import os
import re
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# =================================================
# 기본 UI
# =================================================
st.set_page_config(layout="wide")
st.title("📘 자동차보험 약관 RAG")

# =================================================
# 세션 상태 (버튼 토글 방지)
# =================================================
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False
if "show_original" not in st.session_state:
    st.session_state.show_original = False

# =================================================
# 스트리밍 핸들러
# =================================================
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# =================================================
# PDF 로드
# =================================================
def load_pdf(uploaded_file):
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return PyPDFLoader(path).load()

# =================================================
# 정규식
# =================================================
ARTICLE_START_RE = re.compile(r"(제\s*\d+\s*조\s*\([^)]+\))")
STOP_RE = re.compile(r"(제\s*\d+편|제\s*\d+장|제\s*\d+절)")

# =================================================
# 목차 파싱 (페이지 정확화)
# =================================================
def parse_toc(pages: List[Document]) -> dict:
    toc = {}
    for page in pages[:5]:
        for line in page.page_content.splitlines():
            m = re.search(r"(제\s*\d+조).*?(\d+)$", line.strip())
            if m:
                toc[m.group(1).replace(" ", "")] = int(m.group(2))
    return toc

# =================================================
# 조항 파싱
# =================================================
def parse_articles(pages: List[Document]) -> List[Document]:
    docs = []
    buffer = ""
    current_article = None
    current_page = None

    for page in pages:
        page_no = page.metadata.get("page", 0) + 1
        for line in page.page_content.splitlines():
            if ARTICLE_START_RE.match(line):
                if current_article:
                    docs.append(Document(
                        page_content=buffer.strip(),
                        metadata={"article": current_article, "page": current_page}
                    ))
                current_article = line.strip()
                current_page = page_no
                buffer = line
                continue

            if current_article and STOP_RE.match(line):
                docs.append(Document(
                    page_content=buffer.strip(),
                    metadata={"article": current_article, "page": current_page}
                ))
                current_article = None
                buffer = ""
                continue

            if current_article:
                buffer += "\n" + line

    if current_article and buffer.strip():
        docs.append(Document(
            page_content=buffer.strip(),
            metadata={"article": current_article, "page": current_page}
        ))

    return docs

# =================================================
# 조항 텍스트 정제
# =================================================
def clean_article_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if re.match(r"제\s*\d+편|제\s*\d+장|제\s*\d+절", line.strip()):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()

# =================================================
# 원문 렌더링 (형식 유지)
# =================================================
def render_original_text(text: str):
    for line in text.splitlines():
        line = line.rstrip()
        if re.match(r"[①②③④⑤⑥⑦⑧⑨]", line.strip()):
            st.markdown(f"**{line}**")
        elif re.match(r"\d+\.|[가-하]\.", line.strip()):
            st.markdown(f"&nbsp;&nbsp;{line}", unsafe_allow_html=True)
        else:
            st.markdown(line)

# =================================================
# Chroma DB
# =================================================
def build_db(docs: List[Document], version: str) -> Chroma:
    persist_dir = f"./chroma_db/{version}"
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="insurance_terms",
    )

# =================================================
# 사이드바
# =================================================
st.sidebar.header("⚙ 설정")
VERSION = st.sidebar.selectbox("약관 버전 선택", ["2024-01", "2025-01"])

# =================================================
# 메인
# =================================================
uploaded_file = st.file_uploader("📄 자동차보험 약관 PDF 업로드", type=["pdf"])

if uploaded_file:
    pages = load_pdf(uploaded_file)
    toc_map = parse_toc(pages)
    docs = parse_articles(pages)

    for d in docs:
        d.page_content = clean_article_text(d.page_content)
        m = re.search(r"(제\s*\d+조)", d.metadata["article"])
        if m:
            key = m.group(1).replace(" ", "")
            if key in toc_map:
                d.metadata["page"] = toc_map[key]

    db = build_db(docs, VERSION)

    article_map = {}
    for d in docs:
        m = re.search(r"제\s*(\d+)\s*조", d.metadata["article"])
        if m:
            article_map[int(m.group(1))] = d.metadata["article"]

    articles = [article_map[n] for n in sorted(article_map.keys())]
    selected_article = st.sidebar.selectbox("조항 선택", articles)

    question = st.text_input("❓ 질문을 입력하세요")

    if st.button("질문하기"):
        st.session_state.show_answer = True

    if st.button("📄 선택 조항 원문 보기"):
        st.session_state.show_original = True

    # =================================================
    # 질문 결과
    # =================================================
    if st.session_state.show_answer and question:
        chat_box = st.empty()
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=True,
            callbacks=[StreamHandler(chat_box)],
        )
        retriever = db.as_retriever(
            search_kwargs={"filter": {"article": selected_article}}
        )
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        rag_chain.invoke(question)

    # =================================================
    # 원문 보기
    # =================================================
    if st.session_state.show_original:
        st.divider()
        for d in docs:
            if d.metadata["article"] == selected_article:
                page = d.metadata["page"]
                st.markdown(f"## {selected_article} (p.{page})")
                st.link_button(
                    "PDF 해당 페이지로 이동",
                    f"file:///{uploaded_file.name}#page={page}",
                )
                render_original_text(d.page_content)
                break
