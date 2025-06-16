import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import pandas as pd

class TimeDomainTab:
    """
    Time Domain Analysis Section - Analyze filter behavior with time-domain signals
    """
    
    def __init__(self):
        self.signal_types = {
            'impulse': 'Impulse',
            'step': 'Step',
            'sine': 'Sine Wave',
            'square': 'Square Wave'
        }

    def render(self):
        """Main render function for the time domain analysis section"""
        
        # Custom CSS with three shades of baby pink
        st.markdown("""
        <style>
            .time-panel {
                background: linear-gradient(135deg, #FFC1CC 0%, #FFABB6 100%);
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .time-panel h3 {
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
        </style>
        """, unsafe_allow_html=True)
        
        # Single heading
        st.markdown('<div class="section-header">⏱️ Time Domain Analysis</div>', unsafe_allow_html=True)
        
        # Create main layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            self._render_time_panel()
        
        with col2:
            self._render_time_visualization()
        
        # Time domain analysis section
        st.markdown("---")
        self._render_time_analysis()

    def _render_time_panel(self):
        """Render the time domain control panel"""
        
        st.markdown("""
        <div class="time-panel">
            <h3>⏱️ Time Domain Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal type selection
        signal_type = st.selectbox(
            "🎯 Signal Type",
            options=list(self.signal_types.keys()),
            format_func=lambda x: self.signal_types[x],
            help="Choose the input signal for analysis",
            key="signal_type"
        )
        
        # Signal parameters
        st.markdown("### 📈 Signal Parameters")
        fs = st.session_state.sampling_freq
        duration = st.number_input(
            "⏱️ Duration (s)",
            min_value=0.001,
            max_value=10.0,
            value=0.1,
            step=0.001,
            help="Length of the signal in seconds",
            key="duration"
        )
        
        amplitude = st.number_input(
            "🔊 Amplitude",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Peak amplitude of the signal",
            key="amplitude"
        )
        
        if signal_type in ['sine', 'square']:
            frequency = st.number_input(
                "🎵 Frequency (Hz)",
                min_value=1.0,
                max_value=fs/2,
                value=1000.0,
                step=10.0,
                help="Frequency of the periodic signal",
                key="frequency"
            )
        
        # Filter selection
        if st.session_state.designed_filters or st.session_state.current_b is not None:
            filter_options = ["Current Filter"] + list(st.session_state.designed_filters.keys())
            selected_filter = st.selectbox(
                "🗂️ Select Filter",
                options=filter_options,
                help="Choose a filter to apply to the signal",
                key="select_filter"
            )
            if selected_filter != "Current Filter" and st.button("Load Filter", use_container_width=True):
                if self.load_filter(selected_filter):
                    st.success(f"✅ Loaded filter '{selected_filter}'")
                    st.rerun()
        
        # Generate and process signal
        if st.session_state.current_b is not None and st.session_state.current_a is not None:
            if st.button("🚀 Analyze Signal", type="primary", use_container_width=True):
                self._generate_and_process_signal(signal_type, duration, amplitude, 
                                                  frequency if signal_type in ['sine', 'square'] else None, fs)
        
        # Export signal data
        if 'input_signal' in st.session_state:
            st.markdown("### 💾 Export Signal Data")
            df = pd.DataFrame({
                'Time (s)': st.session_state.time_vector,
                'Input Signal': st.session_state.input_signal,
                'Output Signal': st.session_state.output_signal
            })
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="time_domain_signals.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_control_panel"
            )

    def _render_time_visualization(self):
        """Render the time domain visualization panel"""
        
        st.markdown("""
        <div class="viz-card">
            <h3 style='color: #FFC1CC; text-align: center;'>📊 Time Domain Visualization</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if 'input_signal' in st.session_state:
            time = st.session_state.time_vector
            input_signal = st.session_state.input_signal
            output_signal = st.session_state.output_signal
            
            # Create subplot for input and output signals
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Input Signal', 'Output Signal'),
                vertical_spacing=0.1
            )
            
            # Input signal
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=input_signal,
                    name='Input',
                    line=dict(color='#FFC1CC', width=2)
                ),
                row=1, col=1
            )
            
            # Output signal
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=output_signal,
                    name='Output',
                    line=dict(color='#FFB6C1', width=2)
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                template="plotly_white",
                showlegend=True,
                font=dict(size=12)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.markdown("""
            <div class="viz-card" style='text-align: center; padding: 2rem;'>
                <h3 style='color: #FFC1CC;'>🎯 No Signal Generated</h3>
                <p style='color: #555; font-size: 1rem;'>
                    Configure signal parameters and click "Analyze Signal" to visualize time-domain responses
                </p>
            </div>
            """, unsafe_allow_html=True)

    def _render_time_analysis(self):
        """Render time domain analysis and metrics"""
        
        st.markdown('<div class="section-header">🔍 Time Domain Analysis</div>', unsafe_allow_html=True)
        
        if 'input_signal' in st.session_state:
            input_signal = st.session_state.input_signal
            output_signal = st.session_state.output_signal
            time = st.session_state.time_vector
            fs = st.session_state.sampling_freq
            signal_type = st.session_state.get('signal_type', 'impulse')
            
            # Calculate metrics
            peak_input = np.max(np.abs(input_signal))
            peak_output = np.max(np.abs(output_signal))
            rms_output = np.sqrt(np.mean(output_signal**2))
            
            # Rise time and settling time for step response
            rise_time = None
            settling_time = None
            if signal_type == 'step':
                try:
                    # Normalize output for analysis
                    output_norm = output_signal / np.max(np.abs(output_signal))
                    # Find 10% and 90% points for rise time
                    idx_10 = np.where(output_norm >= 0.1)[0][0]
                    idx_90 = np.where(output_norm >= 0.9)[0][0]
                    rise_time = (idx_90 - idx_10) / fs
                    
                    # Find settling time (within 2% of final value)
                    final_value = output_norm[-1]
                    idx_settle = np.where(np.abs(output_norm - final_value) <= 0.02)[0]
                    if len(idx_settle) > 0:
                        settling_time = idx_settle[0] / fs
                except:
                    pass
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📊 Input Signal Metrics")
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>🔺 Peak Amplitude</h4>
                    <p style='font-size: 1.2rem;'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Maximum absolute amplitude</p>
                </div>
                """.format(peak_input), unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📊 Output Signal Metrics")
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>🔺 Peak Amplitude</h4>
                    <p style='font-size: 1.2rem;'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Maximum absolute amplitude</p>
                </div>
                """.format(peak_output), unsafe_allow_html=True)
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📏 RMS Level</h4>
                    <p style='font-size: 1.2rem;'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Root mean square amplitude</p>
                </div>
                """.format(rms_output), unsafe_allow_html=True)
            
            with col3:
                st.markdown("#### ⚙️ Response Characteristics")
                if rise_time is not None and signal_type == 'step':
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>⏱️ Rise Time</h4>
                        <p style='font-size: 1.2rem;'>{:.6f} s</p>
                        <p style='color: #555; font-size: 0.9rem;'>10% to 90% transition</p>
                    </div>
                    """.format(rise_time), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>⏱️ Rise Time</h4>
                        <p style='font-size: 1.2rem;'>N/A</p>
                        <p style='color: #555; font-size: 0.9rem;'>Available for step response</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if settling_time is not None and signal_type == 'step':
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>⏱️ Settling Time</h4>
                        <p style='font-size: 1.2rem;'>{:.6f} s</p>
                        <p style='color: #555; font-size: 0.9rem;'>Within 2% of final value</p>
                    </div>
                    """.format(settling_time), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>⏱️ Settling Time</h4>
                        <p style='font-size: 1.2rem;'>N/A</p>
                        <p style='color: #555; font-size: 0.9rem;'>Available for step response</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Signal data table
            with st.expander("📋 Signal Data Table"):
                df = pd.DataFrame({
                    'Time (s)': time,
                    'Input Signal': input_signal,
                    'Output Signal': output_signal
                })
                st.dataframe(df.head(50), use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="time_domain_signals.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_analysis_panel"
                )
        
        else:
            st.info("Please configure a signal and select a filter to perform time-domain analysis.")

    def _generate_and_process_signal(self, signal_type, duration, amplitude, frequency, fs):
        """Generate input signal and process it with the selected filter"""
        
        try:
            # Generate time vector
            samples = int(duration * fs)
            time = np.linspace(0, duration, samples)
            
            # Generate input signal
            if signal_type == 'impulse':
                input_signal = signal.unit_impulse(samples, idx=0) * amplitude
            elif signal_type == 'step':
                input_signal = np.ones(samples) * amplitude  # Custom step signal
            elif signal_type == 'sine':
                input_signal = amplitude * np.sin(2 * np.pi * frequency * time)
            elif signal_type == 'square':
                input_signal = amplitude * signal.square(2 * np.pi * frequency * time)
            
            # Apply filter
            b = st.session_state.current_b
            a = st.session_state.current_a
            output_signal = signal.lfilter(b, a, input_signal)
            
            # Store in session state
            st.session_state.input_signal = input_signal
            st.session_state.output_signal = output_signal
            st.session_state.time_vector = time
            
            st.success(f"✅ Signal {self.signal_types[signal_type]} generated and processed successfully!")
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error processing signal: {str(e)}")

    def load_filter(self, filter_name):
        """Load a saved filter for time domain analysis"""
        
        if filter_name in st.session_state.designed_filters:
            filter_data = st.session_state.designed_filters[filter_name]
            st.session_state.current_b = filter_data['b']
            st.session_state.current_a = filter_data['a']
            st.session_state.current_filter = filter_data
            st.session_state.sampling_freq = filter_data['fs']
            return True
        return False
