import streamlit as st
from supabase import create_client
from qdrant_client import QdrantClient
from datasets import load_dataset
from google import genai

# ============================================================
# KHỞI TẠO
# ============================================================
@st.cache_resource
def init_clients():
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
    qdrant = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_KEY"]
    )
    gemini = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    return supabase, qdrant, gemini

@st.cache_resource
def init_model():
    from FlagEmbedding import FlagModel
    return FlagModel("BAAI/bge-m3", use_fp16=True)

@st.cache_resource
def load_hf_dataset():
    return load_dataset(
        st.secrets["HF_USERNAME"] + "/luat-vn-chunks",
        token=st.secrets["HF_TOKEN"],
        split="train"
    )

def get_chunk_text(chunk_id, dataset):
    results = dataset.filter(lambda x: x["chunk_id"] == chunk_id)
    if len(results) > 0:
        return results[0]["text"]
    return None

# ============================================================
# GIAO DIỆN
# ============================================================
st.set_page_config(
    page_title="Hỏi đáp Pháp luật Việt Nam",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Hỏi đáp Pháp luật Việt Nam")
st.caption("Hệ thống hỗ trợ tra cứu và giải đáp pháp luật tự động")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Nhập câu hỏi pháp lý của bạn..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang tìm kiếm..."):

                supabase, qdrant, gemini = init_clients()
                model = init_model()
                dataset = load_hf_dataset()

                # Dịch sang tiếng Việt nếu cần
                detect_prompt = f"Câu này có phải tiếng Việt không? Nếu không, dịch sang tiếng Việt. Chỉ trả về bản tiếng Việt, không giải thích: {query}"
                viet_query = gemini.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=detect_prompt
                ).text.strip()

                # Encode + search Qdrant
                query_vector = model.encode(viet_query)
                if isinstance(query_vector, dict):
                    query_vector = query_vector["dense_vecs"]

                results = qdrant.query_points(
                    collection_name="luat_vn",
                    query=query_vector.tolist(),
                    limit=5
                ).points

                # Lấy metadata + text
                context_parts = []
                sources = []

                for r in results:
                    chunk_id = r.payload["chunk_id"]
                    text = get_chunk_text(chunk_id, dataset)
                    meta = supabase.table("legal_chunks").select("*, legal_documents(*)").eq("chunk_id", chunk_id).execute()

                    if text and meta.data:
                        dieu = meta.data[0]["dieu_so"]
                        van_ban = meta.data[0]["legal_documents"]["title"]
                        context_parts.append(f"{van_ban} - {dieu}:\n{text}")
                        sources.append(f"**{dieu}** — {van_ban}")

                context = "\n\n".join(context_parts)

                # Xử lý khi không tìm được kết quả
                if not context_parts:
                    answer = "Xin lỗi, tôi chưa tìm được điều luật liên quan đến câu hỏi này trong cơ sở dữ liệu hiện tại. Vui lòng thử lại với câu hỏi khác hoặc liên hệ chuyên gia pháp lý."
                else:
                    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Dựa vào các điều luật sau, hãy trả lời câu hỏi một cách rõ ràng, chính xác bằng tiếng Việt.

Các điều luật liên quan:
{context}

Câu hỏi: {viet_query}

Yêu cầu:
- Trả lời trực tiếp, rõ ràng
- Trích dẫn điều luật cụ thể
- Nếu không đủ thông tin, hãy nói rõ"""

                    answer = gemini.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    ).text

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Nguồn tham khảo"):
                        for s in sources:
                            st.markdown(f"- {s}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            st.error(f"Lỗi chi tiết:\n```\n{error_detail}\n```")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Lỗi: {str(e)}"
            })
