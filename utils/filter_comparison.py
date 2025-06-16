import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class FilterComparisonTab:
    """
    Filter Comparison Section - Compare multiple filters in frequency and time domains
    with enhanced baby pink theme and additional features
    """
    
    def __init__(self):
        # Initialize session state if not present
        if 'designed_filters' not in st.session_state:
            st.session_state.designed_filters = {}
        if 'comparison_triggered' not in st.session_state:
            st.session_state.comparison_triggered = False
        if 'comparison_settings' not in st.session_state:
            st.session_state.comparison_settings = {}

        self.analysis_types = {
            'frequency': 'Frequency Response',
            'time': 'Time Domain Response'
        }
        self.response_types = {
            'magnitude': 'Magnitude',
            'phase': 'Phase',
            'impulse': 'Impulse Response',
            'step': 'Step Response'
        }
        # Distinct colors for different filters
        self.colors = ['#66C2A5', '#8DA0CB', '#FC8D62']  # Soft teal, lavender, coral

    def render(self):
        """Main render function for the filter comparison section"""
        
        # Custom CSS with baby pink theme
        st.markdown("""
        <style>
            .comp-panel {
                background: linear-gradient(135deg, #FFC1CC 0%, #FFB6C1 100%);
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(255, 173, 173, 0.2);
                margin-bottom: 2rem;
            }
            .comp-panel h3 {
                color: #333;
                text-align: center;
                font-size: 1.6rem;
                font-family: 'Arial', sans-serif;
                margin-bottom: 1rem;
            }
            .viz-card {
                background-color: #FFF5F5;
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(255, 173, 173, 0.15);
                margin-bottom: 2rem;
                transition: transform 0.3s ease;
            }
            .viz-card:hover {
                transform: translateY(-3px);
            }
            .metric-card {
                background-color: #FFF5F5;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(255, 173, 173, 0.1);
                transition: transform 0.3s ease;
                margin-bottom: 1rem;
            }
            .metric-card:hover {
                transform: translateY(-5px);
            }
            .section-header {
                font-size: 2rem;
                font-weight: bold;
                color: #FFB6C1;
                text-align: center;
                margin-bottom: 2rem;
                font-family: 'Arial', sans-serif;
            }
            .stButton>button {
                background-color: #FFADAD;
                color: white;
                border-radius: 8px;
                font-family: 'Arial', sans-serif;
                transition: background-color 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #FFC1CC;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">🔍 Filter Comparison Dashboard</div>', unsafe_allow_html=True)
        
        # Create main layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            self._render_comparison_panel()
        
        with col2:
            self._render_comparison_visualization()
        
        # Filter metrics section
        st.markdown("---")
        self._render_filter_metrics()

    def _render_comparison_panel(self):
        """Render the comparison control panel"""
        
        st.markdown("""
        <div class="comp-panel">
            <h3>🔧 Comparison Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "🎯 Analysis Type",
            options=list(self.analysis_types.keys()),
            format_func=lambda x: self.analysis_types[x],
            help="Choose between frequency or time domain comparison",
            key="analysis_type"
        )
        
        # Response type selection
        if analysis_type == 'frequency':
            response_type = st.selectbox(
                "📊 Response Type",
                options=['magnitude', 'phase'],
                format_func=lambda x: self.response_types[x],
                help="Choose the response to compare",
                key="response_type"
            )
        else:
            response_type = st.selectbox(
                "📊 Response Type",
                options=['impulse', 'step'],
                format_func=lambda x: self.response_types[x],
                help="Choose the time-domain response to compare",
                key="response_type"
            )
        
        # Filter selection
        if st.session_state.designed_filters:
            selected_filters = st.multiselect(
                "🗂️ Select Filters",
                options=list(st.session_state.designed_filters.keys()),
                help="Select up to 3 filters for comparison",
                max_selections=3,
                key="selected_filters"
            )
        else:
            st.warning("⚠️ No saved filters available. Design filters in the 'Filter Design' section.")
            return
        
        # Plot settings
        st.markdown("### 📏 Plot Settings")
        freq_scale = st.radio(
            "📈 Frequency Scale",
            options=['linear', 'log'],
            format_func=lambda x: 'Linear' if x == 'linear' else 'Logarithmic',
            help="Choose frequency axis scaling (for frequency analysis)",
            key="freq_scale"
        )
        
        if analysis_type == 'time':
            time_duration = st.slider(
                "⏳ Time Duration (s)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Adjust duration for time-domain plots",
                key="time_duration"
            )
        else:
            time_duration = 0.1  # Default value
        
        if analysis_type == 'frequency' and response_type == 'magnitude':
            mag_unit = st.selectbox(
                "📊 Magnitude Units",
                options=['db', 'linear'],
                format_func=lambda x: 'Decibels (dB)' if x == 'db' else 'Linear',
                help="Units for magnitude display",
                key="mag_unit"
            )
        else:
            mag_unit = 'db'
        
        if analysis_type == 'frequency' and response_type == 'phase':
            phase_unit = st.selectbox(
                "🔄 Phase Units",
                options=['degrees', 'radians'],
                format_func=lambda x: 'Degrees' if x == 'degrees' else 'Radians',
                help="Units for phase display",
                key="phase_unit"
            )
        else:
            phase_unit = 'degrees'
        
        with st.expander("⚙️ Advanced Options"):
            show_grid = st.checkbox("Show Grid", value=True, key="show_grid")
            show_legend = st.checkbox("Show Legend", value=True, key="show_legend")
            show_filter_type = st.checkbox("Show Filter Type in Legend", value=True, key="show_filter_type")
        
        # Store settings in session state
        st.session_state.comparison_settings = {
            'analysis_type': analysis_type,
            'response_type': response_type,
            'selected_filters': selected_filters,
            'freq_scale': freq_scale,
            'mag_unit': mag_unit,
            'phase_unit': phase_unit,
            'show_grid': show_grid,
            'show_legend': show_legend,
            'show_filter_type': show_filter_type,
            'time_duration': time_duration
        }
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if selected_filters and st.button("🚀 Compare Filters", type="primary", use_container_width=True):
                st.session_state.comparison_triggered = True
        with col_btn2:
            if st.button("🔄 Reset Settings", use_container_width=True):
                st.session_state.comparison_triggered = False
                st.session_state.comparison_settings = {}
                st.rerun()

    def _render_comparison_visualization(self):
        """Render the comparison visualization panel"""
        
        st.markdown("""
        <div class="viz-card">
            <h3 style='color: #FFB6C1; text-align: center;'>📊 Filter Comparison Visualization</h3>
        </div>
        """, unsafe_allow_html=True)
        
        settings = st.session_state.get('comparison_settings', {})
        if not settings.get('selected_filters') or not st.session_state.get('comparison_triggered'):
            st.markdown("""
            <div class="viz-card" style='text-align: center; padding: 2rem;'>
                <h3 style='color: #FFB6C1;'>🎯 No Filters Selected</h3>
                <p style='color: #555; font-size: 1rem;'>
                    Select filters and click "Compare Filters" to visualize comparisons
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        analysis_type = settings.get('analysis_type', 'frequency')
        response_type = settings.get('response_type', 'magnitude')
        
        try:
            if analysis_type == 'frequency':
                fig = self._create_frequency_comparison_plot(settings)
            else:
                fig = self._create_time_comparison_plot(settings)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add PDF download option
            pdf_buffer = self._generate_pdf_report(fig, settings)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name="filter_comparison_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"⚠️ Error generating visualization: {str(e)}")

    def _create_frequency_comparison_plot(self, settings):
        """Create frequency domain comparison plot"""
        
        selected_filters = settings.get('selected_filters', [])
        freq_scale = settings.get('freq_scale', 'linear')
        mag_unit = settings.get('mag_unit', 'db')
        phase_unit = settings.get('phase_unit', 'degrees')
        show_grid = settings.get('show_grid', True)
        show_legend = settings.get('show_legend', True)
        show_filter_type = settings.get('show_filter_type', True)
        response_type = settings.get('response_type', 'magnitude')
        
        fig = go.Figure()
        
        for idx, filter_name in enumerate(selected_filters):
            filter_data = st.session_state.designed_filters.get(filter_name, {})
            if not filter_data:
                continue
            b = filter_data.get('b', [])
            a = filter_data.get('a', [1])
            fs = filter_data.get('fs', 1000)
            filter_type = filter_data.get('filter_type', 'Unknown')
            
            # Generate frequency vector
            w_hz = np.logspace(np.log10(1), np.log10(fs/2), 8192) if freq_scale == 'log' else np.linspace(1, fs/2, 8192)
            w_rad = 2 * np.pi * w_hz / fs
            try:
                w, h = signal.freqz(b, a, worN=w_rad, fs=fs)
            except Exception:
                continue
            
            if response_type == 'magnitude':
                magnitude = np.abs(h)
                if mag_unit == 'db':
                    y_data = 20 * np.log10(magnitude + 1e-10)
                    y_label = "Magnitude (dB)"
                else:
                    y_data = magnitude
                    y_label = "Magnitude (Linear)"
                
                legend_name = f"{filter_name} ({filter_type})" if show_filter_type else filter_name
                fig.add_trace(go.Scatter(
                    x=w,
                    y=y_data,
                    name=legend_name,
                    line=dict(color=self.colors[idx % len(self.colors)], width=3),
                    showlegend=show_legend
                ))
            
            elif response_type == 'phase':
                phase = np.angle(h)
                if phase_unit == 'degrees':
                    y_data = phase * 180 / np.pi
                    y_label = "Phase (Degrees)"
                else:
                    y_data = phase
                    y_label = "Phase (Radians)"
                
                legend_name = f"{filter_name} ({filter_type})" if show_filter_type else filter_name
                fig.add_trace(go.Scatter(
                    x=w,
                    y=y_data,
                    name=legend_name,
                    line=dict(color=self.colors[idx % len(self.colors)], width=3),
                    showlegend=show_legend
                ))
        
        # Update layout
        fig.update_layout(
            title="",
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_label,
            template="plotly_white",
            height=600,
            showlegend=show_legend,
            font=dict(family="Arial", size=12, color="#333")
        )
        
        if freq_scale == 'log':
            fig.update_xaxes(type="log")
        
        fig.update_layout(
            xaxis=dict(showgrid=show_grid, gridcolor="#E0E0E0"),
            yaxis=dict(showgrid=show_grid, gridcolor="#E0E0E0"),
            plot_bgcolor="#FFF5F5",
            paper_bgcolor="#FFF5F5"
        )
        
        return fig

    def _create_time_comparison_plot(self, settings):
        """Create time domain comparison plot"""
        
        selected_filters = settings.get('selected_filters', [])
        response_type = settings.get('response_type', 'impulse')
        show_grid = settings.get('show_grid', True)
        show_legend = settings.get('show_legend', True)
        show_filter_type = settings.get('show_filter_type', True)
        duration = settings.get('time_duration', 0.1)
        
        fig = go.Figure()
        
        for idx, filter_name in enumerate(selected_filters):
            filter_data = st.session_state.designed_filters.get(filter_name, {})
            if not filter_data:
                continue
            b = filter_data.get('b', [])
            a = filter_data.get('a', [1])
            fs = filter_data.get('fs', 1000)
            filter_type = filter_data.get('filter_type', 'Unknown')
            
            # Generate time vector and input signal
            samples = int(duration * fs)
            time = np.linspace(0, duration, samples)
            if response_type == 'impulse':
                input_signal = signal.unit_impulse(samples)
            elif response_type == 'step':
                input_signal = np.ones(samples)
            
            try:
                output_signal = signal.lfilter(b, a, input_signal)
            except Exception:
                continue
            
            legend_name = f"{filter_name} ({filter_type})" if show_filter_type else filter_name
            fig.add_trace(go.Scatter(
                x=time,
                y=output_signal,
                name=legend_name,
                line=dict(color=self.colors[idx % len(self.colors)], width=3),
                showlegend=show_legend
            ))
        
        # Update layout
        fig.update_layout(
            title="",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=600,
            showlegend=show_legend,
            font=dict(family="Arial", size=12, color="#333")
        )
        
        fig.update_layout(
            xaxis=dict(showgrid=show_grid, gridcolor="#E0E0E0"),
            yaxis=dict(showgrid=show_grid, gridcolor="#E0E0E0"),
            plot_bgcolor="#FFF5F5",
            paper_bgcolor="#FFF5F5"
        )
        
        return fig

    def _render_filter_metrics(self):
        """Render filter comparison metrics"""
        
        st.markdown('<div class="section-header">🔍 Filter Performance Metrics</div>', unsafe_allow_html=True)
        
        settings = st.session_state.get('comparison_settings', {})
        selected_filters = settings.get('selected_filters', [])
        
        if not selected_filters:
            st.info("Please select filters to compare their metrics.")
            return
        
        cols = st.columns(min(len(selected_filters), 3))
        
        for idx, filter_name in enumerate(selected_filters):
            with cols[idx % 3]:
                st.markdown(f"#### {filter_name}")
                filter_data = st.session_state.designed_filters.get(filter_name, {})
                if not filter_data:
                    st.error(f"No data available for {filter_name}")
                    continue
                b = filter_data.get('b', [])
                a = filter_data.get('a', [1])
                fs = filter_data.get('fs', 1000)
                
                try:
                    # Calculate metrics
                    order = max(len(b), len(a)) - 1
                    poles = np.roots(a)
                    is_stable = np.all(np.abs(poles) < 1)
                    stability_text = "✅ Stable" if is_stable else "⚠️ Unstable"
                    
                    # Bandwidth calculation with error handling
                    w, h = signal.freqz(b, a, worN=1000, fs=fs)
                    magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
                    peak_mag = np.max(magnitude_db)
                    cutoff_indices = np.where(magnitude_db >= peak_mag - 3)[0]
                    bandwidth = w[cutoff_indices[-1]] - w[cutoff_indices[0]] if len(cutoff_indices) > 1 else None
                    
                    # Display metrics
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFB6C1;'>📈 Filter Order</h4>
                        <p style='font-size: 1.2rem;'>{}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Number of coefficients</p>
                    </div>
                    """.format(order), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFB6C1;'>🔧 Stability</h4>
                        <p style='font-size: 1.2rem;'>{}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Filter stability status</p>
                    </div>
                    """.format(stability_text), unsafe_allow_html=True)
                    
                    if bandwidth is not None:
                        st.markdown("""
                        <div class="metric-card">
                            <h4 style='color: #FFB6C1;'>📏 -3dB Bandwidth</h4>
                            <p style='font-size: 1.2rem;'>{:.2f} Hz</p>
                            <p style='color: #555; font-size: 0.9rem;'>Bandwidth range</p>
                        </div>
                        """.format(bandwidth), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card">
                            <h4 style='color: #FFB6C1;'>📏 -3dB Bandwidth</h4>
                            <p style='font-size: 1.2rem;'>N/A</p>
                            <p style='color: #555; font-size: 0.9rem;'>Unable to calculate</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error calculating metrics for {filter_name}: {str(e)}")
        
        # Comparison data table
        with st.expander("📋 Comparison Data Table"):
            if settings.get('analysis_type') == 'frequency':
                try:
                    freqs = np.logspace(np.log10(1), np.log10(st.session_state.designed_filters[selected_filters[0]].get('fs', 1000)/2), 20)
                    df_data = pd.DataFrame({'Frequency (Hz)': freqs})
                    
                    for filter_name in selected_filters:
                        filter_data = st.session_state.designed_filters.get(filter_name, {})
                        if not filter_data:
                            continue
                        b = filter_data.get('b', [])
                        a = filter_data.get('a', [1])
                        fs = filter_data.get('fs', 1000)
                        _, h = signal.freqz(b, a, worN=2*np.pi*freqs/fs, fs=fs)
                        
                        if settings.get('response_type') == 'magnitude':
                            df_data[f"{filter_name} Magnitude (dB)"] = 20 * np.log10(np.abs(h) + 1e-10)
                        else:
                            df_data[f"{filter_name} Phase (degrees)"] = np.angle(h) * 180 / np.pi
                    
                    st.dataframe(df_data, use_container_width=True)
                    csv = df_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="filter_comparison_frequency.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating frequency data table: {str(e)}")
            
            else:  # time domain
                try:
                    duration = settings.get('time_duration', 0.1)
                    fs = st.session_state.designed_filters[selected_filters[0]].get('fs', 1000)
                    samples = int(duration * fs)
                    time = np.linspace(0, duration, samples)
                    df_data = pd.DataFrame({'Time (s)': time})
                    
                    for filter_name in selected_filters:
                        filter_data = st.session_state.designed_filters.get(filter_name, {})
                        if not filter_data:
                            continue
                        b = filter_data.get('b', [])
                        a = filter_data.get('a', [1])
                        input_signal = signal.unit_impulse(samples) if settings.get('response_type') == 'impulse' else np.ones(samples)
                        output_signal = signal.lfilter(b, a, input_signal)
                        
                        df_data[f"{filter_name} Response"] = output_signal
                    
                    st.dataframe(df_data, use_container_width=True)
                    csv = df_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="filter_comparison_time.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating time data table: {str(e)}")

    def _generate_pdf_report(self, fig, settings):
        """Generate a PDF report of the comparison plot and metrics"""
        
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.setFillColorRGB(1, 0.71, 0.76)  # Baby pink
        c.drawCentredString(width/2, height - 50, "Filter Comparison Report")
        
        # Save plot as image
        try:
            fig.write_png("temp_plot.png", width=500, height=300)
            c.drawImage("temp_plot.png", 50, height - 350, width=500, height=250)
        except Exception:
            c.setFont("Helvetica", 12)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(50, height - 350, "Plot image unavailable")
        
        # Add metrics
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        y_offset = height - 400
        selected_filters = settings.get('selected_filters', [])
        
        for filter_name in selected_filters:
            filter_data = st.session_state.designed_filters.get(filter_name, {})
            if not filter_data:
                continue
            b = filter_data.get('b', [])
            a = filter_data.get('a', [1])
            fs = filter_data.get('fs', 1000)
            
            try:
                order = max(len(b), len(a)) - 1
                poles = np.roots(a)
                is_stable = np.all(np.abs(poles) < 1)
                stability_text = "Stable" if is_stable else "Unstable"
                
                w, h = signal.freqz(b, a, worN=1000, fs=fs)
                magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
                peak_mag = np.max(magnitude_db)
                cutoff_indices = np.where(magnitude_db >= peak_mag - 3)[0]
                bandwidth = f"{(w[cutoff_indices[-1]] - w[cutoff_indices[0]]):.2f} Hz" if len(cutoff_indices) > 1 else "N/A"
                
                c.drawString(50, y_offset, f"Filter: {filter_name}")
                c.drawString(100, y_offset - 20, f"Order: {order}")
                c.drawString(100, y_offset - 40, f"Stability: {stability_text}")
                c.drawString(100, y_offset - 60, f"-3dB Bandwidth: {bandwidth}")
                y_offset -= 80
            except Exception:
                c.drawString(50, y_offset, f"Metrics unavailable for {filter_name}")
                y_offset -= 20
        
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer
