import streamlit as st

def render_home():
    """Render the Home page for Digital Filter Designer with a polished design"""

    # Custom CSS with three shades of baby pink
    st.markdown("""
    <style>
        .hero-section {
            background: linear-gradient(135deg, #FFC1CC 0%, #FFABB6 100%);
            padding: 3rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .hero-section h1 {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .hero-section p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .feature-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        .feature-card h3 {
            color: #FFC1CC;
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
        }
        .feature-card p {
            color: #555;
            font-size: 1rem;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #FFC1CC;
            text-align: center;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🎛️ Welcome to Digital Filter Designer</h1>
        <p>Design, analyze, and visualize digital filters with ease and precision.</p>
    </div>
    """, unsafe_allow_html=True)

    # Introduction to Digital Filters
    st.markdown('<div class="section-header">What Are Digital Filters?</div>', unsafe_allow_html=True)
    st.write("""
    Digital filters are mathematical algorithms used to manipulate digital signals. Whether you're cleaning up noisy audio, 
    extracting important features from biomedical signals, or improving the clarity of communications, digital filters play a key role.
    
    They come in various types such as FIR (Finite Impulse Response) and IIR (Infinite Impulse Response), each suited to specific applications. 
    The goal of this tool is to simplify filter creation and offer hands-on exploration of how filters behave.
    
    Use the sidebar to dive into designing your own filters or testing them on real signals.
    """)

    # Features Section
    st.markdown("---")
    st.markdown('<div class="section-header">Why Choose Digital Filter Designer?</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>Interactive Design</h3>
            <p>Create custom filters with real-time parameter adjustments and visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Comprehensive Analysis</h3>
            <p>Explore frequency response, time domain, and filter coefficients with detailed plots.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>Audio Processing</h3>
            <p>Apply your filters to audio signals and hear the results instantly.</p>
        </div>
        """, unsafe_allow_html=True)

    # Applications Section
    st.markdown("---")
    st.markdown('<div class="section-header">Applications of Digital Filters</div>', unsafe_allow_html=True)
    st.write("""
    Digital filters are essential tools in modern engineering and science. Here are just a few fields where they shine:
    
    - 🎧 **Audio Engineering**: Remove noise, equalize sound, or create special effects.
    - 📡 **Telecommunications**: Improve data transmission quality and remove interference.
    - ❤️ **Biomedical Engineering**: Filter physiological signals like ECG or EEG for diagnosis.
    - 🛰️ **Radar and Satellite Systems**: Enhance signal clarity and extract features.
    - 🧠 **Machine Learning**: Pre-process time-series or speech data for better learning outcomes.
    
    You’ll find dedicated sections for filter design, frequency domain visualization, and real-time audio testing via the sidebar.
    """)

    # Extra Motivation
    st.markdown("---")
    st.markdown('<div class="section-header">👩‍💻 Start Exploring Now!</div>', unsafe_allow_html=True)
    st.write("""
    Whether you're a student, researcher, or audio enthusiast, this platform is your interactive playground for understanding and applying digital filters.
    
    🔍 Navigate through the sidebar to:
    - Experiment with **FIR** and **IIR** filters.
    - Visualize **impulse and frequency responses**.
    - Test your filters on **real audio files** or **synthetic signals**.

    Ready to bring theory to life? Jump into the **Filter Design** section and see signal processing in action!
    """)
