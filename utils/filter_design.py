import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
import pandas as pd

class FilterDesignTab:
    """
    Digital Filter Design Section - Interactive filter design with real-time visualization
    Supports both IIR and FIR filters
    """
    
    def __init__(self):
        self.filter_types = {
            'lowpass': 'Low-pass',
            'highpass': 'High-pass', 
            'bandpass': 'Band-pass',
            'bandstop': 'Band-stop'
        }
        
        self.design_methods = {
            'butter': 'Butterworth (IIR)',
            'cheby1': 'Chebyshev I (IIR)',
            'cheby2': 'Chebyshev II (IIR)',
            'ellip': 'Elliptic (IIR)',
            'bessel': 'Bessel (IIR)',
            'window': 'Window-based (FIR)'
        }
        
        self.window_types = {
            'hamming': 'Hamming',
            'hann': 'Hann',
            'blackman': 'Blackman',
            'kaiser': 'Kaiser',
            'bartlett': 'Bartlett'
        }

    def render(self):
        """Main render function for the filter design section"""
        # Initialize session state variables
        self._initialize_session_state()
        
        # Custom CSS with three shades of baby pink
        st.markdown("""
        <style>
            .design-panel {
                background: linear-gradient(135deg, #FFC1CC 0%, #FFABB6 100%);
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .design-panel h3 {
                color: white;
                text-align: center;
                font-size: 1.5rem;
                margin-bottom: 1rem;
            }
            .viz-card {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }
            .metric-card:hover {
                transform: translateY(-5px);
            }
            .section-header {
                font-size: 1.8rem;
                font-weight: bold;
                color: #FFC1CC;
                text-align: center;
                margin-bottom: 2rem;
            }
            .stButton>button {
                background-color: #FFC1CC;
                color: white;
                border-radius: 5px;
                transition: background-color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #FFB6C1;
            }
            .plotly-trace {
                stroke: #FFABB6 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">🔧 Filter Design</div>', unsafe_allow_html=True)
        
        # Create main layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._render_design_panel()
        
        with col2:
            self._render_visualization_panel()
        
        # Filter performance metrics
        st.markdown("---")
        self._render_performance_metrics()

    def _initialize_session_state(self):
        """Initialize session state variables with safe defaults"""
        if 'window_type' not in st.session_state:
            st.session_state.window_type = 'hamming'
        elif not isinstance(st.session_state.window_type, str) or st.session_state.window_type not in self.window_types:
            st.session_state.window_type = 'hamming'
            
        if 'sampling_freq' not in st.session_state:
            st.session_state.sampling_freq = 44100.0
            
        if 'designed_filters' not in st.session_state:
            st.session_state.designed_filters = {}
            
        if 'filter_counter' not in st.session_state:
            st.session_state.filter_counter = 0

    def _render_design_panel(self):
        """Render the filter design control panel"""
        st.markdown("""
        <div class="design-panel">
            <h3>🔧 Filter Design Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Design method selection
        design_method = st.selectbox(
            "🎯 Design Method",
            options=list(self.design_methods.keys()),
            format_func=lambda x: self.design_methods[x],
            help="Choose the filter design algorithm (IIR or FIR)",
            key="design_method"
        )
        
        # Filter type selection
        filter_type = st.selectbox(
            "📊 Filter Type", 
            options=list(self.filter_types.keys()),
            format_func=lambda x: self.filter_types[x],
            help="Select the type of frequency response",
            key="filter_type"
        )
        
        # Sampling frequency
        st.number_input(
            "🔄 Sampling Frequency (Hz)",
            min_value=1000.0,
            max_value=192000.0,
            value=44100.0,
            step=1000.0,
            help="The sampling rate of your digital system",
            key="sampling_freq"
        )
        
        # Filter order
        order = st.slider(
            "📈 Filter Order",
            min_value=1,
            max_value=100 if design_method == 'window' else 20,
            value=100 if design_method == 'window' else 4,
            help="Higher order = steeper transition, more computation (FIR: typically higher order)",
            key="filter_order"
        )
        
        # Frequency specifications
        st.markdown("### 🎵 Frequency Specifications")
        
        if filter_type in ['lowpass', 'highpass']:
            cutoff_freq = st.number_input(
                f"🎯 Cutoff Frequency (Hz)",
                min_value=1.0,
                max_value=st.session_state.sampling_freq/2,
                value=st.session_state.sampling_freq/8,
                step=100.0,
                help="The -3dB frequency point",
                key="cutoff_freq"
            )
            critical_freqs = [cutoff_freq]
            
        else:  # bandpass or bandstop
            col_low, col_high = st.columns(2)
            with col_low:
                low_freq = st.number_input(
                    "🔽 Low Frequency (Hz)",
                    min_value=1.0,
                    max_value=st.session_state.sampling_freq/2-100,
                    value=5512.5,
                    step=100.0,
                    key="low_freq"
                )
            with col_high:
                high_freq = st.number_input(
                    "🔼 High Frequency (Hz)",
                    min_value=low_freq + 100,
                    max_value=st.session_state.sampling_freq/2,
                    value=11025.0,
                    step=100.0,
                    key="high_freq"
                )
            critical_freqs = [low_freq, high_freq]
        
        # Additional parameters for specific methods
        self._render_method_specific_params(design_method)
        
        # Design button
        if st.button("🚀 Design Filter", type="primary", use_container_width=True):
            self._design_filter(design_method, filter_type, order, critical_freqs)
        
        # Save filter section
        if st.session_state.get('current_b') is not None:
            st.markdown("### 💾 Save Filter")
            filter_name = st.text_input(
                "Filter Name",
                value=f"{self.filter_types[filter_type]}_{self.design_methods[design_method]}_{order}",
                help="Give your filter a memorable name",
                key="filter_name"
            )
            
            if st.button("💾 Save Filter", use_container_width=True):
                self._save_filter(filter_name, design_method, filter_type, order, critical_freqs)

    def _render_method_specific_params(self, method):
        """Render additional parameters for the design method"""
        if method in ['cheby1', 'ellip']:
            st.number_input(
                "🌊 Passband Ripple (dB)",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Maximum allowed ripple in the passband",
                key="passband_ripple"
            )
        
        if method in ['cheby2', 'ellip']:
            st.number_input(
                "🔇 Stopband Attenuation (dB)",
                min_value=20.0,
                max_value=100.0,
                value=40.0,
                step=5.0,
                help="Minimum attenuation in the stopband",
                key="stopband_atten"
            )
        
        if method == 'window':
            st.selectbox(
                "🪗 Window Type",
                options=list(self.window_types.keys()),
                format_func=lambda x: self.window_types[x],
                help="Select window function for FIR filter design",
                key="window_type"
            )
            if st.session_state.get('window_type') == 'kaiser':
                st.number_input(
                    "🔷 Kaiser Beta",
                    min_value=0.0,
                    max_value=14.0,
                    value=6.0,
                    step=0.1,
                    help="Shape parameter for Kaiser window",
                    key="kaiser_beta"
                )

    def _render_visualization_panel(self):
        """Render the real-time visualization panel"""
        st.markdown("""
        <div class="viz-card">
            <h3 style='color: #FFC1CC; text-align: center;'>📊 Filter Analysis Dashboard</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get('current_b') is not None and st.session_state.get('current_a') is not None:
            # Create subplot figure
            fig = make_subplots(
                rows=2, 
                cols=2,
                subplot_titles=('Magnitude Response', 'Phase Response', 
                              'Group Delay', 'Pole-Zero Plot'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Calculate frequency response
            w, h = signal.freqz(st.session_state.current_b, st.session_state.current_a, 
                              worN=8000, fs=st.session_state.sampling_freq)
            
            # Magnitude response
            magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
            fig.add_trace(
                go.Scatter(x=w, y=magnitude_db, name='Magnitude (dB)', 
                          line=dict(color='#FFC1CC', width=3)),
                row=1, col=1
            )
            
            # Phase response
            phase_unwrapped = np.unwrap(np.angle(h)) * 180 / np.pi
            fig.add_trace(
                go.Scatter(x=w, y=phase_unwrapped, name='Phase (degrees)', 
                          line=dict(color='#FFB6C1', width=2)),
                row=1, col=2
            )
            
            # Group delay
            try:
                w_gd, gd = signal.group_delay((st.session_state.current_b, st.session_state.current_a), 
                                            fs=st.session_state.sampling_freq)
                fig.add_trace(
                    go.Scatter(x=w_gd, y=gd, name='Group Delay (samples)', 
                              line=dict(color='#FFADAD', width=2)),
                    row=2, col=1
                )
            except Exception as e:
                fig.add_trace(
                    go.Scatter(x=[], y=[], name='Group Delay (error)', 
                              line=dict(color='#FFADAD', width=2)),
                    row=2, col=1
                )
            
            # Pole-zero plot
            poles = np.roots(st.session_state.current_a)
            zeros = np.roots(st.session_state.current_b)
            
            # Unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            unit_circle_x = np.cos(theta)
            unit_circle_y = np.sin(theta)
            
            fig.add_trace(
                go.Scatter(x=unit_circle_x, y=unit_circle_y, 
                          mode='lines', name='Unit Circle',
                          line=dict(color='gray', dash='dash')),
                row=2, col=2
            )
            
            # Add poles
            if len(poles) > 0:
                fig.add_trace(
                    go.Scatter(x=poles.real, y=poles.imag, 
                              mode='markers', name='Poles',
                              marker=dict(symbol='x', size=12, color='#FFB6C1')),
                    row=2, col=2
                )
            
            # Add zeros
            if len(zeros) > 0:
                fig.add_trace(
                    go.Scatter(x=zeros.real, y=zeros.imag, 
                              mode='markers', name='Zeros',
                              marker=dict(symbol='circle-open', size=10, color='#FFADAD')),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                template="plotly_white",
                font=dict(size=12)
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
            fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
            fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
            fig.update_yaxes(title_text="Phase (degrees)", row=1, col=2)
            fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
            fig.update_yaxes(title_text="Group Delay (samples)", row=2, col=1)
            fig.update_xaxes(title_text="Real Part", row=2, col=2)
            fig.update_yaxes(title_text="Imaginary Part", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display filter coefficients
            with st.expander("🔍 Filter Coefficients"):
                col_b, col_a = st.columns(2)
                with col_b:
                    st.markdown("**Numerator (b) coefficients:**")
                    st.code(f"b = {st.session_state.current_b}", language="python")
                with col_a:
                    st.markdown("**Denominator (a) coefficients:**")
                    st.code(f"a = {st.session_state.current_a}", language="python")
            
        else:
            # Placeholder visualization
            st.markdown("""
            <div class="viz-card" style='text-align: center; padding: 2rem;'>
                <h4 style='color: #FFC1CC;'>🎯 Ready to Design!</h4>
                <p style='color: #555; font-size: 1rem;'>
                    Configure your filter parameters and click "Design Filter" 
                    to see real-time visualizations
                </p>
            </div>
            """, unsafe_allow_html=True)

    def _render_performance_metrics(self):
        """Render filter performance metrics"""
        st.markdown('<h3 style="color: #FFC1CC; text-align: center;">📈 Filter Performance Metrics</h3>', unsafe_allow_html=True)
        
        if st.session_state.get('current_b') is not None and st.session_state.get('current_a') is not None:
            # Calculate metrics
            w, h = signal.freqz(st.session_state.current_b, st.session_state.current_a, 
                              worN=8000, fs=st.session_state.sampling_freq)
            magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
            
            # Find key frequencies and metrics
            passband_ripple = np.max(magnitude_db[:len(magnitude_db)//4]) - np.min(magnitude_db[:len(magnitude_db)//4])
            stopband_atten = -np.min(magnitude_db[3*len(magnitude_db)//4:])
            filter_order = len(st.session_state.current_b) - 1
            
            # Check stability
            poles = np.roots(st.session_state.current_a)
            is_stable = np.all(np.abs(poles) <= 1.0) if len(poles) > 0 else True
            stability = "✅ Stable" if is_stable else "⚠ Unstable"
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>🌟 Passband Ripple</h4>
                        <p style='font-size: 1.2rem;'>{passband_ripple:.2f} dB</p>
                        <p style='color: #555; font-size: 0.9rem;'>Variation in passband</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>🔇 Stopband Attenuation</h4>
                        <p style='font-size: 1.2rem;'>{stopband_atten:.1f} dB</p>
                        <p style='color: #555; font-size: 0.9rem;'>Minimum suppression in stopband</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>📏 Filter Order</h4>
                        <p style='font-size: 1.2rem;'>{filter_order}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Number of filter coefficients</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>🔧 Stability</h4>
                        <p style='font-size: 1.2rem;'>{stability}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Filter stability status</p>
                    </div>
                    """, unsafe_allow_html=True)

    def _design_filter(self, method, filter_type, order, critical_freqs):
        """
        Design the digital filter with specified parameters
        Supports both IIR and FIR methods
        """
        try:
            # Normalize frequencies
            nyquist = st.session_state.sampling_freq / 2
            normalized_freqs = [f / nyquist for f in critical_freqs]
            
            # Validate frequency range
            for freq in normalized_freqs:
                if freq >= 1.0 or freq <= 0.0:
                    raise ValueError(f"Frequency must be between 0 and Nyquist frequency ({nyquist:.1f} Hz)")
            
            # Get and validate window_type
            window_type = st.session_state.get('window_type', 'hamming')
            
            # Validate window_type
            if not isinstance(window_type, str) or window_type not in self.window_types:
                st.warning(f"Invalid window type: {window_type}. Resetting to 'hamming'.")
                window_type = 'hamming'
                st.session_state.window_type = window_type
            
            # Design filter based on method
            if method == 'window':
                # Prepare window parameter for firwin
                if window_type == 'kaiser':
                    beta = st.session_state.get('kaiser_beta', 6.0)
                    window_param = ('kaiser', beta)
                else:
                    window_param = window_type
                
                # Design FIR filter
                if filter_type in ['lowpass', 'highpass']:
                    b = signal.firwin(order + 1, normalized_freqs[0], window=window_param, 
                                    pass_zero=(filter_type == 'lowpass'))
                else:  # bandpass or bandstop
                    b = signal.firwin(order + 1, normalized_freqs, window=window_param, 
                                    pass_zero=(filter_type == 'bandstop'))
                a = [1]  # FIR filters have no feedback coefficients
                
            else:
                # IIR filter design
                rp = st.session_state.get('passband_ripple', 1.0)
                rs = st.session_state.get('stopband_atten', 40.0)
                
                if method == 'butter':
                    b, a = signal.butter(order, normalized_freqs, btype=filter_type)
                elif method == 'cheby1':
                    b, a = signal.cheby1(order, rp, normalized_freqs, btype=filter_type)
                elif method == 'cheby2':
                    b, a = signal.cheby2(order, rs, normalized_freqs, btype=filter_type)
                elif method == 'ellip':
                    b, a = signal.ellip(order, rp, rs, normalized_freqs, btype=filter_type)
                elif method == 'bessel':
                    b, a = signal.bessel(order, normalized_freqs, btype=filter_type)
            
            # Store in session state
            st.session_state.current_b = b
            st.session_state.current_a = a
            st.session_state.current_filter = {
                'method': method,
                'type': filter_type,
                'order': order,
                'frequencies': critical_freqs,
                'fs': st.session_state.sampling_freq,
                'b': b.copy(),
                'a': a.copy() if isinstance(a, np.ndarray) else a,
                'window_type': window_type if method == 'window' else None,
                'kaiser_beta': st.session_state.get('kaiser_beta', None) if method == 'window' and window_type == 'kaiser' else None
            }
            
            st.success(f"✅ {self.design_methods[method]} {self.filter_types[filter_type]} filter designed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error designing filter: {str(e)}")
            st.write(f"Debug info: method={method}, filter_type={filter_type}, order={order}, freqs={critical_freqs}")

    def _save_filter(self, name, method, filter_type, order, critical_freqs):
        """Save the current filter to session state"""
        if not name.strip():
            st.error("Please provide a filter name")
            return
        
        if name in st.session_state.get('designed_filters', {}):
            if not st.checkbox(f"Overwrite existing filter '{name}'?", key=f"overwrite_{name}"):
                return
        
        window_type = st.session_state.get('window_type', 'hamming')
        if method == 'window' and (not isinstance(window_type, str) or window_type not in self.window_types):
            window_type = 'hamming'
            st.session_state.window_type = 'hamming'
        
        if 'designed_filters' not in st.session_state:
            st.session_state.designed_filters = {}
        
        st.session_state.designed_filters[name] = {
            'method': method,
            'type': filter_type,
            'order': order,
            'frequencies': critical_freqs,
            'fs': st.session_state.sampling_freq,
            'b': st.session_state.current_b.copy(),
            'a': st.session_state.current_a.copy() if isinstance(st.session_state.current_a, np.ndarray) else st.session_state.current_a,
            'window_type': window_type,
            'kaiser_beta': st.session_state.get('kaiser_beta', None),
            'timestamp': pd.Timestamp.now()
        }
        
        st.session_state.filter_counter = st.session_state.get('filter_counter', 0) + 1
        st.success(f"💾 Filter '{name}' saved successfully!")
        st.rerun()

    def get_current_filter_info(self):
        """Get information about the currently designed filter"""
        if st.session_state.get('current_filter') is not None:
            return st.session_state.current_filter
        return None

    def apply_filter_to_signal(self, signal_data, fs=None):
        """Apply the currently designed filter to a signal"""
        if st.session_state.get('current_b') is None or st.session_state.get('current_a') is None:
            raise ValueError("No filter designed. Please design a filter first.")
        
        # Apply the filter
        filtered_signal = signal.lfilter(st.session_state.current_b, st.session_state.current_a, signal_data)
        
        return filtered_signal
