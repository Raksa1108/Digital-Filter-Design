import streamlit as st
from utils.home import render_home
from utils.filter_design import FilterDesignTab
from utils.frequency_response import FrequencyResponseTab
from utils.filter_coefficients import FilterCoefficientsTab
from utils.time_domain import TimeDomainTab
from utils.audio_processing import AudioProcessingTab
from utils.filter_comparison import FilterComparisonTab

# Page configuration
st.set_page_config(
    page_title="Digital Filter Designer",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'designed_filters' not in st.session_state:
        st.session_state.designed_filters = {}
    if 'current_filter' not in st.session_state:
        st.session_state.current_filter = None
    if 'filter_counter' not in st.session_state:
        st.session_state.filter_counter = 0
    if 'current_b' not in st.session_state:
        st.session_state.current_b = None
    if 'current_a' not in st.session_state:
        st.session_state.current_a = None
    if 'sampling_freq' not in st.session_state:
        st.session_state.sampling_freq = 44100

def main():
    """Main application function"""
    initialize_session_state()
    
    # App header
    st.markdown('<h1 class="main-header">🎛️ Digital Filter Designer</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📱 Navigation")
        section = st.selectbox(
            "Choose a section",
            [
                "Home",
                "Filter Design",
                "Frequency Response",
                "Filter Coefficients",
                "Time Domain",
                "Audio Processing",
                "Filter Comparison"
            ],
            index=0
        )
        
        st.markdown("### 📱 App Info")
        st.info("""
        **Digital Filter Designer**
        
        Design and analyze digital filters with real-time visualization.
        
        **Features:**
        - Interactive filter design
        - Real-time frequency response
        - Audio processing capabilities
        - Filter comparison tools
        """)
        
        if st.session_state.designed_filters:
            st.markdown("### 🗂️ Saved Filters")
            st.write(f"Total filters: {len(st.session_state.designed_filters)}")
            
            if st.button("Clear All Filters", type="secondary"):
                st.session_state.designed_filters = {}
                st.session_state.current_filter = None
                st.session_state.current_b = None
                st.session_state.current_a = None
                st.rerun()

    # Main content based on selected section
    if section == "Home":
        render_home()
    
    elif section == "Filter Design":
        filter_design_tab = FilterDesignTab()
        filter_design_tab.render()
    
    elif section == "Frequency Response":
        if st.session_state.current_b is not None and st.session_state.current_a is not None:
            freq_response_tab = FrequencyResponseTab()
            freq_response_tab.render()
        else:
            st.info("Please design a filter in the 'Filter Design' section first.")
    
    elif section == "Filter Coefficients":
        if st.session_state.current_b is not None and st.session_state.current_a is not None:
            coefficients_tab = FilterCoefficientsTab()
            coefficients_tab.render()
        else:
            st.info("Please design a filter in the 'Filter Design' section first.")
    
    elif section == "Time Domain":
        if st.session_state.current_b is not None and st.session_state.current_a is not None:
            time_domain_tab = TimeDomainTab()
            time_domain_tab.render()
        else:
            st.info("Please design a filter in the 'Filter Design' section first.")
    
    elif section == "Audio Processing":
        audio_tab = AudioProcessingTab()
        audio_tab.render()
    
    elif section == "Filter Comparison":
        if st.session_state.designed_filters:
            comparison_tab = FilterComparisonTab()
            comparison_tab.render()
        else:
            st.info("Please design at least one filter to use comparison features.")

if __name__ == "__main__":
    main()
