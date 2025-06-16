import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
import pandas as pd

class FrequencyResponseTab:
    """
    Frequency Response Analysis Section - Advanced frequency domain visualization and analysis
    """
    
    def __init__(self):
        self.analysis_types = {
            'magnitude': 'Magnitude Response',
            'phase': 'Phase Response',
            'group_delay': 'Group Delay',
            'combined': 'Combined Analysis',
            'custom': 'Custom Analysis'
        }
        
        self.magnitude_units = {
            'db': 'Decibels (dB)',
            'linear': 'Linear Scale',
            'normalized': 'Normalized'
        }
        
        self.phase_units = {
            'degrees': 'Degrees',
            'radians': 'Radians',
            'unwrapped': 'Unwrapped (degrees)'
        }

    def render(self):
        """Main render function for the frequency response section"""
        
        # Custom CSS with baby pink color scheme
        st.markdown("""
        <style>
            .analysis-panel {
                background: linear-gradient(135deg, #FFC1CC 0%, #FFABB6 100%);
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .analysis-panel h3 {
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
        
        st.markdown('<div class="section-header">📊 Frequency Response Analysis</div>', unsafe_allow_html=True)
        
        # Create main layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            self._render_analysis_panel()
        
        with col2:
            self._render_response_visualization()
        
        # Additional analysis section
        st.markdown("---")
        self._render_detailed_analysis()

    def _render_analysis_panel(self):
        """Render the analysis control panel"""
        
        st.markdown("""
        <div class="analysis-panel">
            <h3>📊 Analysis Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "🎯 Analysis Type",
            options=list(self.analysis_types.keys()),
            format_func=lambda x: self.analysis_types[x],
            help="Choose the type of frequency response analysis",
            key="freq_analysis_type"
        )
        
        # Frequency range settings
        st.markdown("### 🎵 Frequency Range")
        
        fs = st.session_state.sampling_freq
        
        col_freq1, col_freq2 = st.columns(2)
        with col_freq1:
            freq_min = st.number_input(
                "🔽 Min Frequency (Hz)",
                min_value=0.1,
                max_value=fs/2,
                value=1.0,
                step=1.0,
                help="Minimum frequency for analysis",
                key="freq_min"
            )
        
        with col_freq2:
            freq_max = st.number_input(
                "🔼 Max Frequency (Hz)",
                min_value=freq_min + 1,
                max_value=fs/2,
                value=min(fs/2, 20000.0),
                step=1.0,
                help="Maximum frequency for analysis",
                key="freq_max"
            )
        
        # Resolution settings
        resolution_points = st.slider(
            "🔬 Frequency Resolution",
            min_value=512,
            max_value=16384,
            value=8192,
            step=512,
            help="Number of frequency points for analysis",
            key="resolution_points"
        )
        
        # Scale settings
        st.markdown("### 📏 Display Settings")
        
        freq_scale = st.radio(
            "📈 Frequency Scale",
            options=['linear', 'log'],
            format_func=lambda x: 'Linear' if x == 'linear' else 'Logarithmic',
            help="Choose frequency axis scaling",
            key="freq_scale"
        )
        
        if analysis_type in ['magnitude', 'combined', 'custom']:
            mag_unit = st.selectbox(
                "📊 Magnitude Units",
                options=list(self.magnitude_units.keys()),
                format_func=lambda x: self.magnitude_units[x],
                help="Units for magnitude display",
                key="mag_unit"
            )
        
        if analysis_type in ['phase', 'combined', 'custom']:
            phase_unit = st.selectbox(
                "🔄 Phase Units",
                options=list(self.phase_units.keys()),
                format_func=lambda x: self.phase_units[x],
                help="Units for phase display",
                key="phase_unit"
            )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            show_grid = st.checkbox("Show Grid", value=True, key="show_grid")
            show_markers = st.checkbox("Show Data Points", value=False, key="show_markers")
            
            if analysis_type == 'magnitude':
                show_3db = st.checkbox("Mark -3dB Points", value=True, key="show_3db")
                show_6db = st.checkbox("Mark -6dB Points", value=False, key="show_6db")
            
            if analysis_type in ['combined', 'custom']:
                show_envelope = st.checkbox("Show Magnitude Envelope", value=False, key="show_envelope")
        
        # Store settings in session state for use in visualization
        st.session_state.freq_analysis_settings = {
            'analysis_type': analysis_type,
            'freq_min': freq_min,
            'freq_max': freq_max,
            'resolution': resolution_points,
            'freq_scale': freq_scale,
            'mag_unit': mag_unit if analysis_type in ['magnitude', 'combined', 'custom'] else 'db',
            'phase_unit': phase_unit if analysis_type in ['phase', 'combined', 'custom'] else 'degrees',
            'show_grid': show_grid,
            'show_markers': show_markers,
            'show_3db': show_3db if analysis_type == 'magnitude' else False,
            'show_6db': show_6db if analysis_type == 'magnitude' else False,
            'show_envelope': show_envelope if analysis_type in ['combined', 'custom'] else False
        }

    def _render_response_visualization(self):
        """Render the main frequency response visualization"""
        
        st.markdown("""
        <div class="viz-card">
            <h3 style='color: #FFC1CC; text-align: center;'>📊 Frequency Response Dashboard</h3>
        </div>
        """, unsafe_allow_html=True)
        
        settings = st.session_state.get('freq_analysis_settings', {})
        analysis_type = settings.get('analysis_type', 'combined')
        
        # Calculate frequency response
        if st.session_state.current_b is None or st.session_state.current_a is None:
            st.error("❌ No filter selected. Please design or load a filter first.")
            return
        
        b = st.session_state.current_b
        a = st.session_state.current_a
        fs = st.session_state.sampling_freq
        
        # Generate frequency vector
        resolution = settings.get('resolution', 8192)
        freq_min = settings.get('freq_min', 1.0)
        freq_max = settings.get('freq_max', fs/2)
        
        if settings.get('freq_scale', 'linear') == 'log':
            w_hz = np.logspace(np.log10(max(freq_min, 0.1)), np.log10(freq_max), resolution)
        else:
            w_hz = np.linspace(freq_min, freq_max, resolution)
        
        # Calculate frequency response
        w_rad = 2 * np.pi * w_hz / fs
        w, h = signal.freqz(b, a, worN=w_rad, fs=fs)
        
        # Create visualization based on analysis type
        try:
            if analysis_type == 'magnitude':
                fig = self._create_magnitude_plot(w, h, settings)
            elif analysis_type == 'phase':
                fig = self._create_phase_plot(w, h, settings)
            elif analysis_type == 'group_delay':
                fig = self._create_group_delay_plot(b, a, w, settings)
            elif analysis_type == 'combined':
                fig = self._create_combined_plot(b, a, w, h, settings)
            else:  # custom
                fig = self._create_custom_plot(b, a, w, h, settings)
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error generating visualization: {str(e)}")

    def _create_magnitude_plot(self, w, h, settings):
        """Create magnitude response plot"""
        
        magnitude = np.abs(h)
        mag_unit = settings.get('mag_unit', 'db')
        
        # Convert magnitude based on units
        if mag_unit == 'db':
            y_data = 20 * np.log10(magnitude + 1e-10)
            y_label = "Magnitude (dB)"
        elif mag_unit == 'linear':
            y_data = magnitude
            y_label = "Magnitude (Linear)"
        else:  # normalized
            y_data = magnitude / np.max(magnitude)
            y_label = "Normalized Magnitude"
        
        fig = go.Figure()
        
        # Main magnitude trace
        fig.add_trace(go.Scatter(
            x=w,
            y=y_data,
            mode='lines' + (' + markers' if settings.get('show_markers', False) else ''),
            name='Magnitude Response',
            line=dict(color='#FFC1CC', width=3),
            marker=dict(size=4) if settings.get('show_markers', False) else None
        ))
        
        # Add -3dB and -6dB lines if requested
        if settings.get('show_3db', False) and mag_unit == 'db':
            fig.add_hline(y=-3, line_dash="dash", line_color="#FFABB6", 
                         annotation_text="-3dB", annotation_position="top right")
        
        if settings.get('show_6db', False) and mag_unit == 'db':
            fig.add_hline(y=-6, line_dash="dash", line_color="#FFB6C1", 
                         annotation_text="-6dB", annotation_position="top right")
        
        # Update layout
        fig.update_layout(
            title="",
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_label,
            template="plotly_white",
            height=600,
            showlegend=True,
            font=dict(size=12)
        )
        
        # Set frequency scale
        if settings.get('freq_scale', 'linear') == 'log':
            fig.update_xaxes(type="log")
        
        # Grid settings
        fig.update_layout(
            xaxis=dict(showgrid=settings.get('show_grid', True)),
            yaxis=dict(showgrid=settings.get('show_grid', True))
        )
        
        return fig

    def _create_phase_plot(self, w, h, settings):
        """Create phase response plot"""
        
        phase = np.angle(h)
        phase_unit = settings.get('phase_unit', 'degrees')
        
        # Convert phase based on units
        if phase_unit == 'degrees':
            y_data = phase * 180 / np.pi
            y_label = "Phase (degrees)"
        elif phase_unit == 'radians':
            y_data = phase
            y_label = "Phase (radians)"
        else:  # unwrapped degrees
            y_data = np.unwrap(phase) * 180 / np.pi
            y_label = "Unwrapped Phase (degrees)"
        
        fig = go.Figure()
        
        # Main phase trace
        fig.add_trace(go.Scatter(
            x=w,
            y=y_data,
            mode='lines' + (' + markers' if settings.get('show_markers', False) else ''),
            name='Phase Response',
            line=dict(color='#FFB6C1', width=3),
            marker=dict(size=4) if settings.get('show_markers', False) else None
        ))
        
        # Update layout
        fig.update_layout(
            title="",
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_label,
            template="plotly_white",
            height=600,
            showlegend=True,
            font=dict(size=12)
        )
        
        # Set frequency scale
        if settings.get('freq_scale', 'linear') == 'log':
            fig.update_xaxes(type="log")
        
        # Grid settings
        fig.update_layout(
            xaxis=dict(showgrid=settings.get('show_grid', True)),
            yaxis=dict(showgrid=settings.get('show_grid', True))
        )
        
        return fig

    def _create_group_delay_plot(self, b, a, w, settings):
        """Create group delay plot"""
        
        try:
            w_gd, gd = signal.group_delay((b, a), w=w, fs=st.session_state.sampling_freq)
            
            fig = go.Figure()
            
            # Group delay trace
            fig.add_trace(go.Scatter(
                x=w_gd,
                y=gd,
                mode='lines' + (' + markers' if settings.get('show_markers', False) else ''),
                name='Group Delay',
                line=dict(color='#FFABB6', width=3),
                marker=dict(size=4) if settings.get('show_markers', False) else None
            ))
            
            # Update layout
            fig.update_layout(
                title="",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Group Delay (samples)",
                template="plotly_white",
                height=600,
                showlegend=True,
                font=dict(size=12)
            )
            
            # Set frequency scale
            if settings.get('freq_scale', 'linear') == 'log':
                fig.update_xaxes(type="log")
            
            # Grid settings
            fig.update_layout(
                xaxis=dict(showgrid=settings.get('show_grid', True)),
                yaxis=dict(showgrid=settings.get('show_grid', True))
            )
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error calculating group delay: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#FFABB6")
            )
            fig.update_layout(
                title="",
                template="plotly_white",
                height=400
            )
        
        return fig

    def _create_combined_plot(self, b, a, w, h, settings):
        """Create combined magnitude and phase plot"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Magnitude Response', 'Phase Response'),
            vertical_spacing=0.1
        )
        
        # Magnitude response
        magnitude = np.abs(h)
        mag_unit = settings.get('mag_unit', 'db')
        
        if mag_unit == 'db':
            mag_data = 20 * np.log10(magnitude + 1e-10)
            mag_label = "Magnitude (dB)"
        elif mag_unit == 'linear':
            mag_data = magnitude
            mag_label = "Magnitude (Linear)"
        else:  # normalized
            mag_data = magnitude / np.max(magnitude)
            mag_label = "Normalized Magnitude"
        
        fig.add_trace(
            go.Scatter(x=w, y=mag_data, name='Magnitude',
                      line=dict(color='#FFC1CC', width=3)),
            row=1, col=1
        )
        
        # Phase response
        phase = np.angle(h)
        phase_unit = settings.get('phase_unit', 'degrees')
        
        if phase_unit == 'degrees':
            phase_data = phase * 180 / np.pi
            phase_label = "Phase (degrees)"
        elif phase_unit == 'radians':
            phase_data = phase
            phase_label = "Phase (radians)"
        else:  # unwrapped degrees
            phase_data = np.unwrap(phase) * 180 / np.pi
            phase_label = "Unwrapped Phase (degrees)"
        
        fig.add_trace(
            go.Scatter(x=w, y=phase_data, name='Phase',
                      line=dict(color='#FFB6C1', width=3)),
            row=2, col=1
        )
        
        # Add envelope if requested
        if settings.get('show_envelope', False):
            envelope = np.maximum.accumulate(mag_data)
            fig.add_trace(
                go.Scatter(x=w, y=envelope, name='Envelope',
                          line=dict(color='#FFABB6', dash='dash', width=2)),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="",
            height=800,
            template="plotly_white",
            showlegend=True,
            font=dict(size=12)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text=mag_label, row=1, col=1)
        fig.update_yaxes(title_text=phase_label, row=2, col=1)
        
        # Set frequency scale
        if settings.get('freq_scale', 'linear') == 'log':
            fig.update_xaxes(type="log")
        
        return fig

    def _create_custom_plot(self, b, a, w, h, settings):
        """Create custom analysis plot with multiple options"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Magnitude Response', 'Phase Response', 
                          'Group Delay', 'Impulse Response'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Magnitude response
        magnitude = np.abs(h)
        mag_data = 20 * np.log10(magnitude + 1e-10)
        
        fig.add_trace(
            go.Scatter(x=w, y=mag_data, name='Magnitude (dB)',
                      line=dict(color='#FFC1CC', width=2)),
            row=1, col=1
        )
        
        # Phase response
        phase_data = np.unwrap(np.angle(h)) * 180 / np.pi
        
        fig.add_trace(
            go.Scatter(x=w, y=phase_data, name='Phase (degrees)',
                      line=dict(color='#FFB6C1', width=2)),
            row=1, col=2
        )
        
        # Group delay
        try:
            w_gd, gd = signal.group_delay((b, a), w=w, fs=st.session_state.sampling_freq)
            fig.add_trace(
                go.Scatter(x=w_gd, y=gd, name='Group Delay',
                          line=dict(color='#FFABB6', width=2)),
                row=2, col=1
            )
        except:
            pass
        
        # Impulse response
        try:
            impulse = signal.unit_impulse(100)
            response = signal.lfilter(b, a, impulse)
            time_samples = np.arange(len(response))
            
            fig.add_trace(
                go.Scatter(x=time_samples, y=response, name='Impulse Response',
                          line=dict(color='#FFABB6', width=2)),
                row=2, col=2
            )
        except:
            pass
        
        # Update layout
        fig.update_layout(
            title="",
            height=800,
            template="plotly_white",
            showlegend=True,
            font=dict(size=10)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_xaxes(title_text="Sample", row=2, col=2)
        
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
        fig.update_yaxes(title_text="Phase (degrees)", row=1, col=2)
        fig.update_yaxes(title_text="Group Delay (samples)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=2)
        
        # Set frequency scale
        if settings.get('freq_scale', 'linear') == 'log':
            fig.update_xaxes(type="log", row=1, col=1)
            fig.update_xaxes(type="log", row=1, col=2)
            fig.update_xaxes(type="log", row=2, col=1)
        
        return fig

    def _render_detailed_analysis(self):
        """Render detailed frequency response analysis and metrics"""
        
        st.markdown('<div class="section-header">🔍 Detailed Analysis</div>', unsafe_allow_html=True)
        
        # Calculate detailed metrics
        if st.session_state.current_b is None or st.session_state.current_a is None:
            st.info("Please design or load a filter to perform detailed analysis.")
            return
        
        b = st.session_state.current_b
        a = st.session_state.current_a
        fs = st.session_state.sampling_freq
        
        # Calculate frequency response
        w, h = signal.freqz(b, a, worN=8192, fs=fs)
        magnitude = np.abs(h)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        phase = np.angle(h)
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Magnitude Metrics")
            
            # Peak magnitude
            peak_mag = np.max(magnitude_db)
            peak_freq = w[np.argmax(magnitude_db)]
            
            st.markdown("""
            <div class="metric-card">
                <h4 style='color: #FFC1CC;'>🔺 Peak Magnitude</h4>
                <p style='font-size: 1.2rem;'>{:.2f} dB</p>
                <p style='color: #555; font-size: 0.9rem;'>@ {:.1f} Hz</p>
            </div>
            """.format(peak_mag, peak_freq), unsafe_allow_html=True)
            
            # DC gain
            dc_gain = magnitude_db[0]
            st.markdown("""
            <div class="metric-card">
                <h4 style='color: #FFC1CC;'>⚡ DC Gain</h4>
                <p style='font-size: 1.2rem;'>{:.2f} dB</p>
                <p style='color: #555; font-size: 0.9rem;'>At 0 Hz</p>
            </div>
            """.format(dc_gain), unsafe_allow_html=True)
            
            # Bandwidth (if applicable)
            try:
                cutoff_indices = np.where(magnitude_db >= peak_mag - 3)[0]
                if len(cutoff_indices) > 1:
                    bandwidth = w[cutoff_indices[-1]] - w[cutoff_indices[0]]
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>📏 -3dB Bandwidth</h4>
                        <p style='font-size: 1.2rem;'>{:.1f} Hz</p>
                        <p style='color: #555; font-size: 0.9rem;'>Bandwidth range</p>
                    </div>
                    """.format(bandwidth), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>📏 -3dB Bandwidth</h4>
                        <p style='font-size: 1.2rem;'>N/A</p>
                        <p style='color: #555; font-size: 0.9rem;'>Insufficient data</p>
                    </div>
                    """, unsafe_allow_html=True)
            except:
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📏 -3dB Bandwidth</h4>
                    <p style='font-size: 1.2rem;'>N/A</p>
                    <p style='color: #555; font-size: 0.9rem;'>Calculation error</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🔄 Phase Metrics")
            
            # Phase at DC
            phase_dc = phase[0] * 180 / np.pi
            st.markdown("""
            <div class="metric-card">
                <h4 style='color: #FFC1CC;'>⚡ Phase at DC</h4>
                <p style='font-size: 1.2rem;'>{:.1f}°</p>
                <p style='color: #555; font-size: 0.9rem;'>At 0 Hz</p>
            </div>
            """.format(phase_dc), unsafe_allow_html=True)
            
            # Phase variation
            phase_range = (np.max(phase) - np.min(phase)) * 180 / np.pi
            st.markdown("""
            <div class="metric-card">
                <h4 style='color: #FFC1CC;'>📊 Phase Range</h4>
                <p style='font-size: 1.2rem;'>{:.1f}°</p>
                <p style='color: #555; font-size: 0.9rem;'>Phase variation</p>
            </div>
            """.format(phase_range), unsafe_allow_html=True)
            
            # Phase linearity
            try:
                phase_unwrapped = np.unwrap(phase)
                phase_linear_fit = np.polyfit(w, phase_unwrapped, 1)
                phase_linear = np.polyval(phase_linear_fit, w)
                phase_deviation = np.std(phase_unwrapped - phase_linear) * 180 / np.pi
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📏 Phase Linearity</h4>
                    <p style='font-size: 1.2rem;'>{:.3f}°</p>
                    <p style='color: #555; font-size: 0.9rem;'>Deviation from linear phase</p>
                </div>
                """.format(phase_deviation), unsafe_allow_html=True)
            except:
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📏 Phase Linearity</h4>
                    <p style='font-size: 1.2rem;'>N/A</p>
                    <p style='color: #555; font-size: 0.9rem;'>Unable to calculate</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### ⚙️ Filter Characteristics")
            
            # Filter order
            filter_order = max(len(b), len(a)) - 1
            st.markdown("""
            <div class="metric-card">
                <h4 style='color: #FFC1CC;'>📈 Filter Order</h4>
                <p style='font-size: 1.2rem;'>{}</p>
                <p style='color: #555; font-size: 0.9rem;'>Number of coefficients</p>
            </div>
            """.format(filter_order), unsafe_allow_html=True)
            
            # Stability check
            poles = np.roots(a)
            is_stable = np.all(np.abs(poles) < 1)
            stability_text = "✅ Stable" if is_stable else "⚠️ Unstable"
            st.markdown("""
            <div class="metric-card">
                <h4 style='color: #FFC1CC;'>🔧 Stability</h4>
                <p style='font-size: 1.2rem;'>{}</p>
                <p style='color: #555; font-size: 0.9rem;'>Filter stability status</p>
            </div>
            """.format(stability_text), unsafe_allow_html=True)
            
            # Minimum phase check
            zeros = np.roots(b)
            is_min_phase = np.all(np.abs(zeros) < 1)
            phase_text = "✅ Min Phase" if is_min_phase else "⚠️ Non-Min Phase"
            st.markdown("""
            <div class="metric-card">
                <h4 style='color: #FFC1CC;'>🔄 Phase Type</h4>
                <p style='font-size: 1.2rem;'>{}</p>
                <p style='color: #555; font-size: 0.9rem;'>Phase characteristic</p>
            </div>
            """.format(phase_text), unsafe_allow_html=True)
        
        # Frequency response table
        with st.expander("📋 Frequency Response Table"):
            # Create frequency points for table
            freq_points = np.logspace(np.log10(max(1.0, fs/1000)), np.log10(fs/2), 20)
            _, h_table = signal.freqz(b, a, worN=2*np.pi*freq_points/fs, fs=fs)
            
            # Create dataframe
            df = pd.DataFrame({
                'Frequency (Hz)': np.round(freq_points, 2),
                'Magnitude (dB)': np.round(20 * np.log10(np.abs(h_table) + 1e-10), 2),
                'Magnitude (Linear)': np.round(np.abs(h_table), 4),
                'Phase (degrees)': np.round(np.angle(h_table) * 180 / np.pi, 2),
                'Phase (radians)': np.round(np.angle(h_table), 4)
            })
            
            st.dataframe(df, use_container_width=True)
            
            # Download button for the table
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="frequency_response.csv",
                mime="text/csv",
                use_container_width=True,
                key="freq_response_download"
            )
