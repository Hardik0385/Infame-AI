import streamlit as st
import random
from g4f.client import Client
st.set_page_config(
    page_title="InfameAI",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_resource
def get_client():
    return Client()

client = get_client()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_processed_input' not in st.session_state:
    st.session_state.last_processed_input = ""
st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
    }
    
    .title {
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        padding: 0.5rem 0;
        font-weight: bold;
    }
    
    .subtitle {
        font-family: 'Arial', sans-serif;
        font-size: 1.1rem;
        color: #cccccc;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .chat-container {
        background-color: #121212;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333333;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .user-message {
        background-color: #1a2b58;
        border-left: 4px solid #3a5cc9;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
        color: #ffffff;
    }
    
    .bot-message {
        background-color: #0f2535;
        border-left: 4px solid #00a3c4;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input {
        background-color: #222222;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 5px;
        padding: 10px;
    }
    
    .stButton > button {
        background-color: #2e4482;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #3a5cc9;
        box-shadow: 0 0 15px rgba(58, 92, 201, 0.5);
    }
    
    .metrics-container {
        background-color: #121212;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333333;
        margin-top: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .metric-item {
        padding: 8px 10px;
        margin: 5px 0;
        border-left: 3px solid #00a3c4;
        background-color: #1a2b3c;
        font-size: 0.9rem;
        color: #dddddd;
    }
    
    .metrics-title {
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid #333333;
    }
    
    .metrics-subtitle {
        color: #00a3c4;
        font-size: 0.9rem;
        text-align: center;
        margin: -5px 0 10px 0;
    }
    
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #aaaaaa;
        font-size: 0.8rem;
        padding: 10px;
        border-top: 1px solid #333333;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    
    .metric-card {
        background-color: #1a2b3c;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 5px 0;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cccccc !important;
    }
    
    label {
        color: #cccccc !important;
    }
    
    hr {
        border-color: #333333 !important;
    }
    
    
    div.stMarkdown p {
        color: #dddddd;
    }
    
    div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def handle_submit():
    input_text = st.session_state.user_text
    if input_text and input_text != st.session_state.last_processed_input:
        st.session_state.last_processed_input = input_text
        st.session_state.messages.append({"role": "user", "content": input_text})
        st.session_state.user_text = ""
        with st.spinner("InfluenceIQ Assistant is thinking..."):
            context = """
            InfluenceIQ is an AI-powered system that ranks who really matters in the digital world.
            
            Key Features:
            - Measures credibility & trustworthiness in a person's field
            - Tracks fame longevity (how long someone has remained relevant)
            - Analyzes meaningful engagement (positive influence vs. just trending)
            - Creates fair ratings that distinguish consistent achievers from short-term viral hits
            
            Key Challenges InfluenceIQ Addresses:
            - Fighting fake fame by preventing spam reviews and manipulative ratings
            - Balancing recent buzz with established legacy when measuring influence
            - Providing real-time ratings that adapt with trends while staying credible
            - Using AI responsibly to ensure unbiased results with ethical considerations
            
            Benefits:
            - Bridges the fame gap by going beyond just followers and likes
            - Uses AI for fair evaluation of genuine influence
            - Provides value to recruiters, brands, and researchers seeking credible collaborators
            - Offers a dynamic rating system that evolves with changing times
            """
        
            prompt = f"User asked about InfluenceIQ: '{input_text}'\n\nInformation about InfluenceIQ:\n{context}\n\nProvide a helpful, conversational response to the user's query as the InfluenceIQ chatbot. Keep responses concise, informative, and engaging."
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    web_search=False
                )
                bot_response = response.choices[0].message.content
            except Exception as e:
                bot_response = f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

st.markdown("<h1 class='title'>InfluenceIQ</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>The AI-Powered System That Ranks Who Really Matters!</p>", unsafe_allow_html=True)

st.markdown("<hr style='margin: 0.5rem 0; border-color: #333333;'>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            """
            <div class='bot-message'>
                <strong>InfluenceIQ Assistant:</strong> Hi! I'm your InfluenceIQ assistant. 
                How can I help you understand our AI-powered influence ranking system?
            </div>
            """, 
            unsafe_allow_html=True
        )

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='bot-message'><strong>InfluenceIQ Assistant:</strong> {message['content']}</div>", 
                unsafe_allow_html=True
            )
    with st.form(key="chat_form", clear_on_submit=True):
        if "user_text" not in st.session_state:
            st.session_state.user_text = ""
        
        st.text_input(
            "Ask about InfluenceIQ:", 
            key="user_text", 
            placeholder="e.g., How does InfluenceIQ measure credibility?"
        )
        
        submit_button = st.form_submit_button(
            label="Send", 
            on_click=handle_submit
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
        <div class="metrics-container">
            <h3 class="metrics-title">Key Metrics</h3>
            <p class="metrics-subtitle">What we measure to determine influence</p>
            <div class="metric-item">⭐ Credibility & Trustworthiness</div>
            <div class="metric-item">⏳ Fame Longevity</div>
            <div class="metric-item">📈 Meaningful Engagement</div>
            <div class="metric-item">🔒 Anti-Manipulation Systems</div>
            <div class="metric-item">⚖️ Buzz vs. Legacy Balance</div>
            <div class="metric-item">🌐 Real-Time Adaptation</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="metrics-container">
            <h3 class="metrics-title">Sample Influence Rating</h3>
            <p class="metrics-subtitle">Live example of our scoring system</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    if 'random_data' not in st.session_state:
        st.session_state.random_data = [random.randint(60, 95) for _ in range(3)]
    st.markdown('<div style="padding: 0 10px;">', unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Credibility", f"{st.session_state.random_data[0]}%", "+5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_b:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Longevity", f"{st.session_state.random_data[1]}%", "+2%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_c:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Engagement", f"{st.session_state.random_data[2]}%", "-3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.button(
        "Generate New Rating", 
        key="new_rating",
        on_click=lambda: setattr(st.session_state, 'random_data', [random.randint(60, 95) for _ in range(3)])
    )

    st.markdown(
        """
        <div class="metrics-container">
            <h3 class="metrics-title">Quick Tips</h3>
            <p class="metrics-subtitle">Improve your influence score</p>
            <div class="metric-item">✅ Post consistent, quality content</div>
            <div class="metric-item">✅ Engage authentically with your audience</div>
            <div class="metric-item">✅ Build connections with industry experts</div>
            <div class="metric-item">✅ Share valuable insights in your field</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="footer">
        InfluenceIQ © 2025 | Redefining fame—fairly, intelligently, and transparently
    </div>
    """, 
    unsafe_allow_html=True
)