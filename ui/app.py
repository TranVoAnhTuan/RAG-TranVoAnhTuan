import streamlit as st
import requests
import uuid
from streamlit_extras.bottom_container import bottom

# ── Config ──────────────────────────────────────────────────────────────────────
import os
API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000") + "/api/v1"

st.set_page_config(
	page_title="Document Assistant",
	page_icon="⚡",
	layout="wide",
	initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
def inject_css():
	st.markdown("""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

	/* ── Global reset ─────────────────────────────────────────────────────── */
	*, *::before, *::after { box-sizing: border-box; }

	html, body, .stApp {
		font-family: 'Inter', sans-serif;
		background: #0a0a0f;
		color: #e2e8f0;
	}

	/* ── Sidebar ───────────────────────────────────────────────────────────── */
	[data-testid="stSidebar"] {
		background: #0f0f18 !important;
		border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
		padding: 0 !important;
	}
	[data-testid="stSidebar"] > div:first-child {
		padding: 1.25rem 1rem 2rem 1rem;
	}

	/* ── Sidebar buttons (topics) ──────────────────────────────────────────── */
	div[data-testid="stSidebar"] div.stButton > button {
		width: 100%;
		border: none !important;
		background: transparent !important;
		text-align: left;
		justify-content: flex-start;
		padding: 0.55rem 0.75rem;
		font-size: 0.8rem;
		font-family: 'Inter', sans-serif;
		font-weight: 500;
		border-radius: 8px;
		color: #94a3b8;
		transition: all 0.18s ease;
		cursor: pointer;
		box-shadow: none !important;
	}
	div[data-testid="stSidebar"] div.stButton > button:hover {
		background: rgba(99, 102, 241, 0.12) !important;
		color: #c7d2fe !important;
	}
	div[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
		background: rgba(99, 102, 241, 0.18) !important;
		color: #818cf8 !important;
		border-left: 2px solid #6366f1 !important;
		border-radius: 0 8px 8px 0 !important;
		padding-left: calc(0.75rem - 2px) !important;
	}

	/* ── Main area ─────────────────────────────────────────────────────────── */
	.block-container {
		padding: 3rem 2rem 4rem 2rem !important; /* Increased top padding */
		max-width: 900px;
		margin: 0 auto;
	}

	/* ── Header banner ─────────────────────────────────────────────────────── */
	.document-header {
		background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(168,85,247,0.08) 100%);
		border: 1px solid rgba(99, 102, 241, 0.2);
		border-radius: 16px;
		padding: 1.25rem 1.5rem;
		margin-bottom: 1.5rem;
		display: flex;
		align-items: center;
		gap: 1rem;
		backdrop-filter: blur(10px);
	}
	.document-header-icon {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		border-radius: 12px;
		width: 44px;
		height: 44px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 1.4rem;
		flex-shrink: 0;
	}
	.document-header-title {
		font-size: 1.15rem;
		font-weight: 700;
		background: linear-gradient(90deg, #818cf8, #c084fc);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
		margin: 0;
		line-height: 1.2;
	}
	.document-header-sub {
		font-size: 0.78rem;
		color: #64748b;
		margin: 2px 0 0 0;
	}
	.topic-badge {
		margin-left: auto;
		background: rgba(99, 102, 241, 0.15);
		border: 1px solid rgba(99, 102, 241, 0.3);
		border-radius: 20px;
		padding: 0.3rem 0.85rem;
		font-size: 0.75rem;
		font-weight: 600;
		color: #a5b4fc;
		white-space: nowrap;
	}

	/* ── Empty / welcome state ──────────────────────────────────────────────── */
	.welcome-card {
		text-align: center;
		padding: 3.5rem 2rem;
		background: linear-gradient(135deg, rgba(15,15,24,0.8), rgba(20,20,35,0.8));
		border: 1px solid rgba(99, 102, 241, 0.12);
		border-radius: 20px;
		margin: 1rem 0 2rem 0;
	}
	.welcome-orb {
		width: 72px;
		height: 72px;
		background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
		border-radius: 50%;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		font-size: 2rem;
		margin-bottom: 1.25rem;
		box-shadow: 0 0 40px rgba(99,102,241,0.35);
		animation: pulse-glow 3s ease-in-out infinite;
	}
	@keyframes pulse-glow {
		0%, 100% { box-shadow: 0 0 30px rgba(99,102,241,0.3); }
		50%       { box-shadow: 0 0 55px rgba(99,102,241,0.55); }
	}
	.welcome-title {
		font-size: 1.5rem;
		font-weight: 700;
		color: #e2e8f0;
		margin: 0 0 0.5rem 0;
	}
	.welcome-sub {
		font-size: 0.875rem;
		color: #64748b;
		margin: 0 0 1.5rem 0;
		line-height: 1.6;
		text-align: center;
	}

	/* ── Suggestion chips ───────────────────────────────────────────────────── */
	div.stButton.chip > button {
		background: rgba(99, 102, 241, 0.08) !important;
		border: 1px solid rgba(99, 102, 241, 0.2) !important;
		border-radius: 100px !important;
		padding: 0.45rem 1rem !important;
		font-size: 0.78rem !important;
		color: #a5b4fc !important;
		transition: all 0.18s ease !important;
		font-family: 'Inter', sans-serif !important;
	}
	div.stButton.chip > button:hover {
		background: rgba(99, 102, 241, 0.18) !important;
		border-color: rgba(99, 102, 241, 0.4) !important;
		color: #c7d2fe !important;
		transform: translateY(-1px);
	}

	/* ── Chat bubbles ───────────────────────────────────────────────────────── */
	.user-bubble {
		background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(139,92,246,0.12));
		border: 1px solid rgba(99, 102, 241, 0.2);
		padding: 0.75rem 1.1rem;
		border-radius: 18px 18px 4px 18px;
		max-width: 82%;
		margin-left: auto;
		font-size: 0.875rem;
		line-height: 1.6;
		color: #e2e8f0;
	}
	.assistant-row {
		display: flex;
		gap: 12px;
		align-items: flex-start;
		max-width: 88%;
	}
	.assistant-avatar {
		background: linear-gradient(135deg, #6366f1, #8b5cf6);
		border-radius: 50%;
		width: 30px;
		height: 30px;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		font-size: 0.9rem;
		margin-top: 2px;
		box-shadow: 0 0 12px rgba(99,102,241,0.4);
	}
	.assistant-content {
		flex-grow: 1;
		font-size: 0.875rem;
		line-height: 1.7;
		color: #cbd5e1;
		background: rgba(15,15,24,0.6);
		border: 1px solid rgba(99,102,241,0.1);
		padding: 0.8rem 1.1rem;
		border-radius: 4px 18px 18px 18px;
	}
	.user-row {
		display: flex;
		justify-content: flex-end;
		margin-bottom: 0.5rem;
	}

	/* ── Citations ──────────────────────────────────────────────────────────── */
	.citation-card {
		background: rgba(15,15,24,0.7);
		border: 1px solid rgba(99,102,241,0.15);
		border-radius: 10px;
		padding: 0.65rem 0.9rem;
		margin-bottom: 0.5rem;
		font-size: 0.78rem;
		line-height: 1.5;
	}
	.citation-h1 { color: #a5b4fc; font-weight: 600; }
	.citation-h2 { color: #64748b; margin-top: 2px; }
	.citation-link {
		color: #818cf8;
		text-decoration: none;
		font-size: 0.73rem;
		display: inline-flex;
		align-items: center;
		gap: 4px;
		margin-top: 4px;
	}
	.citation-link:hover { color: #c7d2fe; }

	/* ── Expander ───────────────────────────────────────────────────────────── */
	details > summary {
		font-size: 0.75rem !important;
		font-weight: 600 !important;
		color: #64748b !important;
	}
	[data-testid="stExpander"] {
		background: transparent !important;
		border: none !important;
	}

	/* ── Chat input ─────────────────────────────────────────────────────────── */
	[data-testid="stChatInput"] {
		background: rgba(15,15,24,0.95) !important;
		border: 1px solid rgba(99,102,241,0.25) !important;
		border-radius: 14px !important;
		backdrop-filter: blur(10px);
	}
	[data-testid="stChatInput"]:focus-within {
		border-color: rgba(99,102,241,0.5) !important;
		box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
	}

	/* ── Sidebar text inputs ────────────────────────────────────────────────── */
	div[data-testid="stSidebar"] input[type="text"] {
		background: rgba(99,102,241,0.06) !important;
		border: 1px solid rgba(99,102,241,0.2) !important;
		border-radius: 8px !important;
		color: #e2e8f0 !important;
		font-size: 0.8rem !important;
		font-family: 'Inter', sans-serif !important;
	}

	/* ── File uploader ──────────────────────────────────────────────────────── */
	[data-testid="stFileUploadDropzone"] {
		background: rgba(99,102,241,0.04) !important;
		border: 1px dashed rgba(99,102,241,0.25) !important;
		border-radius: 10px !important;
	}

	/* ── Divider ────────────────────────────────────────────────────────────── */
	hr { border-color: rgba(99,102,241,0.12) !important; }

	/* ── Spinner ────────────────────────────────────────────────────────────── */
	.stSpinner > div { border-top-color: #6366f1 !important; }

	/* ── Hide default Streamlit chrome ──────────────────────────────────────── */
	#MainMenu, footer { visibility: hidden; } /* Removed header to allow opening sidebar */
	[data-testid="stChatMessage"] > div:first-child { display: none; }
	[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0 !important; }

	/* ── Scrollbar ──────────────────────────────────────────────────────────── */
	::-webkit-scrollbar { width: 5px; }
	::-webkit-scrollbar-track { background: transparent; }
	::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 10px; }
	::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }
	</style>
	""", unsafe_allow_html=True)


# ── API helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_topics() -> list[str]:
	try:
		res = requests.get(f"{API_URL}/topics", timeout=5)
		if res.status_code == 200:
			return res.json().get("topics", ["General"])
	except Exception:
		pass
	return ["General"]


# ── Session state ─────────────────────────────────────────────────────────────────
def init_state():
	if "thread_id" not in st.session_state:
		st.session_state.thread_id = str(uuid.uuid4())
	if "messages" not in st.session_state:
		st.session_state.messages = []
	if "available_topics" not in st.session_state:
		st.session_state.available_topics = fetch_topics()
	else:
		# Merge fresh topics from backend without overwriting locally added ones
		fresh_topics = fetch_topics()
		for t in fresh_topics:
			if t not in st.session_state.available_topics:
				st.session_state.available_topics.append(t)
	if "active_topic" not in st.session_state:
		topics = st.session_state.available_topics
		st.session_state.active_topic = topics[0] if topics else "General"
	if "trigger_send" not in st.session_state:
		st.session_state.trigger_send = None
	if "last_upload_status" not in st.session_state:
		st.session_state.last_upload_status = None
	if "pending_interrupt" not in st.session_state:
		st.session_state.pending_interrupt = None


# ── Sidebar ───────────────────────────────────────────────────────────────────────
def render_sidebar():
	with st.sidebar:
		# Logo
		st.markdown("""
		<div style="display:flex;align-items:center;gap:10px;padding:0.5rem 0 1.25rem 0;border-bottom:1px solid rgba(99,102,241,0.15);margin-bottom:1.25rem;">
			<div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:10px;width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;box-shadow:0 0 16px rgba(99,102,241,0.4);">⚡</div>
			<div>
				<div style="font-weight:700;font-size:1rem;background:linear-gradient(90deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">DocumentAssistant</div>
				<div style="font-size:0.65rem;color:#475569;margin-top:1px;">AI Knowledge Base</div>
			</div>
		</div>
		""", unsafe_allow_html=True)

		# Topics
		st.markdown("<p style='font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;margin-bottom:0.5rem;'>Knowledge Topics</p>", unsafe_allow_html=True)

		for topic in st.session_state.available_topics:
			is_active = topic == st.session_state.active_topic
			label = f"{'📂' if is_active else '📁'} {topic}"
			btn_type = "primary" if is_active else "secondary"
			if st.button(label, key=f"topic_{topic}", type=btn_type, use_container_width=True):
				if not is_active:
					st.session_state.active_topic = topic
					st.session_state.messages = []
					st.rerun()

		# New topic input
		st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
		new_topic = st.text_input(
			"new_topic", placeholder="+ New topic…",
			label_visibility="collapsed", key="new_topic_input"
		)
		if new_topic and new_topic.strip() and new_topic.strip() not in st.session_state.available_topics:
			st.session_state.available_topics.append(new_topic.strip())
			st.session_state.active_topic = new_topic.strip()
			st.rerun()

		st.divider()
		
		# Upload
		st.markdown(f"<p style='font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#475569;margin-bottom:0.5rem;'>Index Document → {st.session_state.active_topic}</p>", unsafe_allow_html=True)
		
		# Show last upload status if exists
		if st.session_state.last_upload_status:
			status_type, status_msg = st.session_state.last_upload_status
			if status_type == "success":
				st.success(status_msg)
			else:
				st.warning(status_msg)
			# Clear it after showing so it doesn't stay forever if they navigate around, 
			# but it will stay for this rerun.
			st.session_state.last_upload_status = None
		uploaded_file = st.file_uploader(
			"upload", type=["pdf"], label_visibility="collapsed",
			key="file_uploader"
		)
		if uploaded_file:
			if st.button("🚀 Process & Index", use_container_width=True, type="primary"):
				with st.spinner("Indexing document…"):
					try:
						res = requests.post(
							f"{API_URL}/upload",
							files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
							data={"topic": st.session_state.active_topic},
							timeout=300,
						)
						if res.status_code == 200:
							result = res.json()
							msg = result.get("message", "Done!")
							
							if "Skipped" in msg or "already exists" in msg:
								st.session_state.last_upload_status = ("warning", msg)
							else:
								st.session_state.last_upload_status = ("success", msg)
								
							st.session_state.available_topics = fetch_topics()
							st.rerun()
						else:
							st.error(f"Error: {res.text}")
					except requests.exceptions.ConnectionError:
						st.error("Cannot connect to backend.")

		st.divider()

		# Clear chat
		if st.button("🧹 Clear Conversation", use_container_width=True):
			st.session_state.messages = []
			st.session_state.thread_id = str(uuid.uuid4())
			st.rerun()

		# Status indicator
		st.markdown(f"""
		<div style="margin-top:auto;padding-top:1rem;font-size:0.68rem;color:#334155;display:flex;align-items:center;gap:6px;">
			<span style="width:7px;height:7px;background:#22c55e;border-radius:50%;display:inline-block;box-shadow:0 0 6px rgba(34,197,94,0.6);"></span>
			Thread: {st.session_state.thread_id[:8]}…
		</div>
		""", unsafe_allow_html=True)


# ── Chat area ─────────────────────────────────────────────────────────────────────
def render_chat():
	active = st.session_state.active_topic

	# ── Empty state ──────────────────────────────────────────────────────────────
	if not st.session_state.messages:
		st.markdown(f"""
		<div class="welcome-card">
			<div class="welcome-orb">⚡</div>
			<div style="margin-bottom: 1.25rem;">
				<span class="topic-badge">📂 {active}</span>
			</div>
			<h2 class="welcome-title">What would you like to know?</h2>
			<p class="welcome-sub">
				Ask anything — I'll search, rerank, and synthesize an accurate answer with citations.
			</p>
		</div>
		""", unsafe_allow_html=True)

		# Suggestion chips
		col1, col2, col3 = st.columns(3)
		chips = [
			("📊 Summarize key findings", "Summarize the key findings of the documents."),
			("🔍 What are the main topics?", "What are the main topics covered in the documents?"),
			("📋 List the key policies", "List the key policies or rules described in the documents."),
		]
		for col, (label, prompt) in zip([col1, col2, col3], chips):
			with col:
				with st.container():
					st.markdown('<div class="chip">', unsafe_allow_html=True)
					if st.button(label, key=f"chip_{label}", use_container_width=True):
						st.session_state.trigger_send = prompt
						st.rerun()
					st.markdown('</div>', unsafe_allow_html=True)

	# ── Conversation history ─────────────────────────────────────────────────────
	for msg in st.session_state.messages:
		with st.chat_message(msg["role"]):
			if msg["role"] == "user":
				st.markdown(
					f'<div class="user-row"><div class="user-bubble">{msg["content"]}</div></div>',
					unsafe_allow_html=True
				)
			else:
				st.markdown(f"""
				<div class="assistant-row">
					<div class="assistant-avatar">⚡</div>
					<div class="assistant-content">{msg["content"]}</div>
				</div>
				""", unsafe_allow_html=True)

				if msg.get("citations"):
					with st.expander(f"📚 {len(msg['citations'])} Source(s)"):
						for i, cite in enumerate(msg["citations"], 1):
							h1 = cite.get("Header_1", "Unknown")
							h2 = cite.get("Header_2", "")
							url = cite.get("file_url", "")
							link_html = f'<a class="citation-link" href="{url}" target="_blank">🔗 View source PDF</a>' if url else ""
							st.markdown(f"""
							<div class="citation-card">
								<div class="citation-h1">{i}. {h1}</div>
								{"<div class='citation-h2'>" + h2 + "</div>" if h2 else ""}
								{link_html}
							</div>
							""", unsafe_allow_html=True)

	# ── Input ────────────────────────────────────────────────────────────────────
	if st.session_state.pending_interrupt:
		st.warning("Please approve or reject the pending action below before asking another question.")
		prompt = None
		user_input = None
	else:
		with bottom():
			user_input = st.chat_input(f"Ask about '{active}'…")

	prompt = user_input or st.session_state.trigger_send
	if not prompt and not st.session_state.pending_interrupt:
		return

	if prompt:
		st.session_state.trigger_send = None
		st.session_state.messages.append({"role": "user", "content": prompt})

		with st.chat_message("user"):
			st.markdown(
				f'<div class="user-row"><div class="user-bubble">{prompt}</div></div>',
				unsafe_allow_html=True
			)

		with st.chat_message("assistant"):
			with st.spinner("Searching knowledge base…"):
				try:
					res = requests.post(
						f"{API_URL}/chat",
						json={
							"query": prompt,
							"thread_id": st.session_state.thread_id,
							"topic": active,
						},
						timeout=300,
					)
					if res.status_code == 200:
						raw = res.json().get("answer", {})
						if isinstance(raw, dict) and raw.get("interrupt"):
							st.session_state.pending_interrupt = raw.get("action_requests", [])
							st.rerun()
						
						if isinstance(raw, dict):
							answer = raw.get("response", "No answer.")
							citations = raw.get("citations", [])
						else:
							answer = str(raw)
							citations = []

						st.markdown(f"""
						<div class="assistant-row">
							<div class="assistant-avatar">⚡</div>
							<div class="assistant-content">{answer}</div>
						</div>
						""", unsafe_allow_html=True)

						if citations:
							with st.expander(f"📚 {len(citations)} Source(s)"):
								for i, cite in enumerate(citations, 1):
									h1 = cite.get("Header_1", "Unknown")
									h2 = cite.get("Header_2", "")
									url = cite.get("file_url", "")
									link_html = f'<a class="citation-link" href="{url}" target="_blank">🔗 View source PDF</a>' if url else ""
									st.markdown(f"""
									<div class="citation-card">
										<div class="citation-h1">{i}. {h1}</div>
										{"<div class='citation-h2'>" + h2 + "</div>" if h2 else ""}
										{link_html}
									</div>
									""", unsafe_allow_html=True)

						st.session_state.messages.append({
							"role": "assistant",
							"content": answer,
							"citations": citations,
						})
					else:
						st.error(f"Backend error {res.status_code}: {res.text}")

				except requests.exceptions.ConnectionError:
					st.error("Cannot connect to the backend. Is FastAPI running?")
				except requests.exceptions.Timeout:
					st.error("Request timed out. The agent may still be processing — try again.")

	# ── Pending Interrupt UI ─────────────────────────────────────────────────────
	def resume_agent(decision):
		st.session_state.pending_interrupt = None
		with st.spinner("Resuming agent..."):
			try:
				res = requests.post(
					f"{API_URL}/chat/resume",
					json={"thread_id": st.session_state.thread_id, "decision": decision},
					timeout=300
				)
				if res.status_code == 200:
					raw = res.json().get("answer", {})
					if isinstance(raw, dict) and raw.get("interrupt"):
						st.session_state.pending_interrupt = raw.get("action_requests", [])
						st.rerun()
					
					if isinstance(raw, dict):
						answer = raw.get("response", "No answer.")
						citations = raw.get("citations", [])
					else:
						answer = str(raw)
						citations = []
					st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
					st.rerun()
				else:
					st.error(f"Backend error {res.status_code}: {res.text}")
			except Exception as e:
				st.error(f"Error resuming: {e}")

	if st.session_state.pending_interrupt:
		st.markdown("### ⚠️ Agent requires your approval")
		for req in st.session_state.pending_interrupt:
			st.info(f"**Tool:** {req.get('name')}\n\n**Arguments:** {req.get('arguments')}")
		col1, col2 = st.columns(2)
		with col1:
			if st.button("✅ Approve", use_container_width=True, type="primary"):
				resume_agent("approve")
		with col2:
			if st.button("❌ Reject", use_container_width=True):
				resume_agent("reject")


# ── Entry point ───────────────────────────────────────────────────────────────────
def main():
	inject_css()
	init_state()
	render_sidebar()
	render_chat()


if __name__ == "__main__":
	main()