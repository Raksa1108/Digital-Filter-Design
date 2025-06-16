import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import soundfile as sf
import io
import librosa
import pandas as pd

class AudioProcessingTab:
    """
    Audio Processing Section - Apply filters to audio signals with visualization, playback, and educational features
    """
    
    def __init__(self):
        self.audio_formats = ['wav', 'mp3', 'flac']
        self.max_file_size_mb = 50  # Limit file size for processing

    def render(self):
        """Main render function for the audio processing section"""
        
        # Custom CSS with baby pink color scheme
        st.markdown("""
        <style>
            .audio-panel {
                background: linear-gradient(135deg, #FFC1CC 0%, #FFABB6 100%);
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .audio-panel h3 {
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
            .learn-more {
                background-color: #FFF5F7;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #FFABB6;
                margin-top: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">🎵 Audio Processing</div>', unsafe_allow_html=True)
        
        # Create main layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            self._render_audio_panel()
        
        with col2:
            self._render_audio_visualization()
        
        # Audio analysis section
        st.markdown("---")
        self._render_audio_analysis()
        
        # Educational section
        self._render_learn_more()

    def _render_audio_panel(self):
        """Render the audio processing control panel"""
        
        st.markdown("""
        <div class="audio-panel">
            <h3>🎵 Audio Processing Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio file upload with tooltip
        st.markdown("""
        <div title="Upload an audio file to process and visualize.\nSupported formats: wav, mp3, flac.\nMax size: 50MB">
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📤 Upload Audio File",
            type=self.audio_formats,
            help=f"Upload an audio file ({', '.join(self.audio_formats)}). Max size: {self.max_file_size_mb}MB",
            key="audio_upload"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                st.error(f"❌ File size ({file_size_mb:.2f}MB) exceeds {self.max_file_size_mb}MB limit.")
                return
            
            # Load audio
            try:
                audio_data, sample_rate = sf.read(io.BytesIO(uploaded_file.read()))
                if len(audio_data.shape) > 1:  # Convert stereo to mono
                    audio_data = np.mean(audio_data, axis=1)
                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = sample_rate
                st.success(f"✅ Audio file loaded successfully! Sample rate: {sample_rate} Hz")
            except Exception as e:
                st.error(f"❌ Error loading audio: {str(e)}")
                return
        
        # Filter selection
        if st.session_state.get('designed_filters', {}) or st.session_state.get('current_b') is not None:
            st.markdown("""
            <div title="Select a designed filter to apply to the audio.\n'Current Filter' uses the active filter coefficients.">
            """, unsafe_allow_html=True)
            filter_options = ["Current Filter"] + list(st.session_state.designed_filters.keys())
            selected_filter = st.selectbox(
                "🗂️ Select Filter",
                options=filter_options,
                help="Choose a filter to apply to the audio",
                key="audio_select_filter"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            if selected_filter != "Current Filter" and st.button("Load Filter", use_container_width=True, key="audio_load_filter"):
                if self.load_filter(selected_filter):
                    st.success(f"✅ Loaded filter '{selected_filter}'")
                    st.rerun()
        
        # Processing parameters
        if st.session_state.get('current_b') is not None and st.session_state.get('current_a') is not None:
            st.markdown("### 🎛️ Processing Parameters")
            st.markdown("""
            <div title="Adjust the gain applied to the filtered audio.\nPositive values amplify, negative values attenuate.">
            """, unsafe_allow_html=True)
            gain = st.slider(
                "🔊 Gain (dB)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                help="Adjust the output gain of the filtered audio",
                key="audio_gain"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            if 'audio_data' in st.session_state and st.button("🚀 Process Audio", type="primary", use_container_width=True, key="process_audio"):
                self._process_audio(gain)
        
        # Export filtered audio
        if 'filtered_audio' in st.session_state:
            st.markdown("### 💾 Export Filtered Audio")
            st.markdown("""
            <div title="Choose the format for the exported filtered audio.\nSupported formats: wav, mp3.">
            """, unsafe_allow_html=True)
            output_format = st.selectbox(
                "📥 Output Format",
                options=['wav', 'mp3'],
                help="Choose the format for the exported audio",
                key="audio_output_format"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            buffer = io.BytesIO()
            sf.write(buffer, st.session_state.filtered_audio, st.session_state.sample_rate, format=output_format.upper())
            st.download_button(
                label="📥 Download Filtered Audio",
                data=buffer.getvalue(),
                file_name=f"filtered_audio.{output_format}",
                mime=f"audio/{output_format}",
                use_container_width=True,
                key="download_filtered_audio"
            )

    def _render_audio_visualization(self):
        """Render the audio visualization panel with waveform and spectrogram"""
        
        st.markdown("""
        <div class="viz-card">
            <h3 style='color: #FFC1CC; text-align: center;'>📊 Audio Visualization</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if 'audio_data' in st.session_state:
            audio_data = st.session_state.audio_data
            sample_rate = st.session_state.sample_rate
            time = np.arange(len(audio_data)) / sample_rate
            
            # Visualization controls
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                show_grid = st.checkbox("Show Grid", value=True, key="audio_show_grid")
            with col_viz2:
                zoom_level = st.slider("Zoom Level", min_value=1, max_value=10, value=1, key="audio_zoom_level")
            
            # Create subplot for waveforms and spectrograms
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original Waveform', 'Filtered Waveform', 'Original Spectrogram', 'Filtered Spectrogram'),
                vertical_spacing=0.1,
                specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "heatmap"}], [{"type": "heatmap"}]]
            )
            
            # Original waveform
            fig.add_trace(
                go.Scatter(
                    x=time[:len(audio_data)//zoom_level],
                    y=audio_data[:len(audio_data)//zoom_level],
                    name='Original',
                    line=dict(color='#FFC1CC', width=2),
                    hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Filtered waveform
            if 'filtered_audio' in st.session_state:
                filtered_audio = st.session_state.filtered_audio
                fig.add_trace(
                    go.Scatter(
                        x=time[:len(filtered_audio)//zoom_level],
                        y=filtered_audio[:len(filtered_audio)//zoom_level],
                        name='Filtered',
                        line=dict(color='#FFB6C1', width=2),
                        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.4f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Spectrograms
            try:
                # Original spectrogram
                D = np.abs(librosa.stft(audio_data))
                D_db = librosa.amplitude_to_db(D, ref=np.max)
                fig.add_trace(
                    go.Heatmap(
                        x=time[:D.shape[1]//zoom_level],
                        y=np.linspace(0, sample_rate/2, D.shape[0]),
                        z=D_db,
                        colorscale='Viridis',
                        name='Original Spectrogram',
                        hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.0f}Hz<br>Amplitude: %{z:.1f}dB<extra></extra>'
                    ),
                    row=3, col=1
                )
                
                # Filtered spectrogram
                if 'filtered_audio' in st.session_state:
                    D_filtered = np.abs(librosa.stft(filtered_audio))
                    D_filtered_db = librosa.amplitude_to_db(D_filtered, ref=np.max)
                    fig.add_trace(
                        go.Heatmap(
                            x=time[:D_filtered.shape[1]//zoom_level],
                            y=np.linspace(0, sample_rate/2, D_filtered.shape[0]),
                            z=D_filtered_db,
                            colorscale='Viridis',
                            name='Filtered Spectrogram',
                            hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.0f}Hz<br>Amplitude: %{z:.1f}dB<extra></extra>'
                        ),
                        row=4, col=1
                    )
            except Exception as e:
                st.warning(f"⚠️ Error computing spectrogram: {str(e)}")
            
            # Update layout
            fig.update_layout(
                height=1200,
                template="plotly_white",
                showlegend=True,
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time (s)", row=1, col=1, showgrid=show_grid)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1, showgrid=show_grid)
            fig.update_xaxes(title_text="Time (s)", row=3, col=1, showgrid=show_grid)
            fig.update_xaxes(title_text="Time (s)", row=4, col=1, showgrid=show_grid)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1, showgrid=show_grid)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1, showgrid=show_grid)
            fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1, showgrid=show_grid)
            fig.update_yaxes(title_text="Frequency (Hz)", row=4, col=1, showgrid=show_grid)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Audio playback
            st.markdown("### 🎧 Audio Playback")
            col_play1, col_play2 = st.columns(2)
            with col_play1:
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, sample_rate, format='WAV')
                st.audio(buffer.getvalue(), format='audio/wav')
                st.markdown("<p style='text-align: center; color: #FFC1CC;'>Original Audio</p>", unsafe_allow_html=True)
            with col_play2:
                if 'filtered_audio' in st.session_state:
                    buffer = io.BytesIO()
                    sf.write(buffer, filtered_audio, sample_rate, format='WAV')
                    st.audio(buffer.getvalue(), format='audio/wav')
                    st.markdown("<p style='text-align: center; color: #FFB6C1;'>Filtered Audio</p>", unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class="viz-card" style='text-align: center; padding: 2rem;'>
                <h3 style='color: #FFC1CC;'>🎯 No Audio Loaded</h3>
                <p style='color: #555; font-size: 1rem;'>
                    Upload an audio file and select a filter to visualize and process audio
                </p>
            </div>
            """, unsafe_allow_html=True)

    def _render_audio_analysis(self):
        """Render audio analysis, metrics, and pole-zero impact"""
        
        st.markdown('<div class="section-header">🔍 Audio Analysis</div>', unsafe_allow_html=True)
        
        if 'audio_data' in st.session_state:
            audio_data = st.session_state.audio_data
            sample_rate = st.session_state.sample_rate
            
            # Calculate metrics
            rms = np.sqrt(np.mean(audio_data**2))
            peak_amplitude = np.max(np.abs(audio_data))
            duration = len(audio_data) / sample_rate
            clipping_risk = peak_amplitude > 0.95  # Warn if close to clipping
            
            if 'filtered_audio' in st.session_state:
                filtered_rms = np.sqrt(np.mean(st.session_state.filtered_audio**2))
                filtered_peak = np.max(np.abs(st.session_state.filtered_audio))
                filtered_clipping_risk = filtered_peak > 0.95
            else:
                filtered_rms = None
                filtered_peak = None
                filtered_clipping_risk = False
            
            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📊 Original Audio Metrics")
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📏 RMS Level</h4>
                    <p style='font-size: 1.2rem;'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Root mean square amplitude</p>
                </div>
                """.format(rms), unsafe_allow_html=True)
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>🔺 Peak Amplitude</h4>
                    <p style='font-size: 1.2rem; color: {}'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Maximum absolute amplitude{}</p>
                </div>
                """.format('#FF0000' if clipping_risk else '#FFC1CC', 
                          peak_amplitude, 
                          '<br><span style="color: #FF0000;">⚠️ Clipping risk!</span>' if clipping_risk else ''), 
                          unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📊 Filtered Audio Metrics")
                if filtered_rms is not None:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>📏 RMS Level</h4>
                        <p style='font-size: 1.2rem;'>{:.6f}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Root mean square amplitude</p>
                    </div>
                    """.format(filtered_rms), unsafe_allow_html=True)
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>🔺 Peak Amplitude</h4>
                        <p style='font-size: 1.2rem; color: {}'>{:.6f}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Maximum absolute amplitude{}</p>
                    </div>
                    """.format('#FF0000' if filtered_clipping_risk else '#FFC1CC', 
                              filtered_peak, 
                              '<br><span style="color: #FF0000;">⚠️ Clipping risk!</span>' if filtered_clipping_risk else ''), 
                              unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>📏 RMS Level</h4>
                        <p style='font-size: 1.2rem;'>N/A</p>
                        <p style='color: #555; font-size: 0.9rem;'>Process audio to compute</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>🔺 Peak Amplitude</h4>
                        <p style='font-size: 1.2rem;'>N/A</p>
                        <p style='color: #555; font-size: 0.9rem;'>Process audio to compute</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("#### ⚙️ Audio Properties")
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>⏱️ Duration</h4>
                    <p style='font-size: 1.2rem;'>{:.2f} s</p>
                    <p style='color: #555; font-size: 0.9rem;'>Total audio length</p>
                </div>
                """.format(duration), unsafe_allow_html=True)
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>🔄 Sample Rate</h4>
                    <p style='font-size: 1.2rem;'>{}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Samples per second</p>
                </div>
                """.format(sample_rate), unsafe_allow_html=True)
            
            # Pole-zero impact analysis
            if st.session_state.get('current_b') is not None and st.session_state.get('current_a') is not None:
                with st.expander("🔍 Filter Pole-Zero Impact"):
                    b = st.session_state.current_b
                    a = st.session_state.current_a
                    zeros = np.roots(b) if len(b) > 1 else np.array([])
                    poles = np.roots(a) if len(a) > 1 else np.array([])
                    is_stable = np.all(np.abs(poles) < 1) if len(poles) > 0 else True
                    
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>🔍 Number of Zeros</h4>
                        <p style='font-size: 1.2rem;'>{}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Roots of numerator polynomial</p>
                    </div>
                    """.format(len(zeros)), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>🔍 Number of Poles</h4>
                        <p style='font-size: 1.2rem;'>{}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Roots of denominator polynomial</p>
                    </div>
                    """.format(len(poles)), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style='color: #FFC1CC;'>🔧 Filter Stability</h4>
                        <p style='font-size: 1.2rem; color: {}'>{}</p>
                        <p style='color: #555; font-size: 0.9rem;'>Affects audio processing reliability</p>
                    </div>
                    """.format('#FF0000' if not is_stable else '#FFC1CC', 
                              "⚠️ Unstable" if not is_stable else "✅ Stable"), 
                              unsafe_allow_html=True)
                    
                    if len(zeros) > 0 or len(poles) > 0:
                        st.markdown("### Pole-Zero Locations")
                        df = pd.DataFrame({
                            'Type': ['Zero'] * len(zeros) + ['Pole'] * len(poles),
                            'Real Part': np.concatenate([zeros.real, poles.real]) if len(zeros) + len(poles) > 0 else [],
                            'Imaginary Part': np.concatenate([zeros.imag, poles.imag]) if len(zeros) + len(poles) > 0 else [],
                            'Magnitude': np.concatenate([np.abs(zeros), np.abs(poles)]) if len(zeros) + len(poles) > 0 else []
                        })
                        st.dataframe(df, use_container_width=True)
            
            # Frequency content analysis
            with st.expander("📋 Frequency Content Table"):
                freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(audio_data)//2]
                fft_magnitude = np.abs(np.fft.fft(audio_data))[:len(audio_data)//2]
                df = pd.DataFrame({
                    'Frequency (Hz)': freqs,
                    'Magnitude (dB)': 20 * np.log10(fft_magnitude + 1e-10)
                })
                st.dataframe(df.head(20), use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="audio_frequency_content.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_freq_content"
                )
        
        else:
            st.info("Please upload an audio file to analyze and process.")

    def _process_audio(self, gain_db):
        """Process the audio with the selected filter and gain"""
        
        try:
            audio_data = st.session_state.audio_data
            b = st.session_state.current_b
            a = st.session_state.current_a
            
            # Apply filter
            filtered_audio = signal.lfilter(b, a, audio_data)
            
            # Apply gain
            gain_linear = 10 ** (gain_db / 20)
            filtered_audio *= gain_linear
            
            # Normalize to prevent clipping
            max_amplitude = np.max(np.abs(filtered_audio))
            if max_amplitude > 1.0:
                filtered_audio /= max_amplitude
                st.warning("⚠️ Audio was normalized to prevent clipping.")
            
            st.session_state.filtered_audio = filtered_audio
            st.success("✅ Audio processed successfully!")
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")

    def load_filter(self, filter_name):
        """Load a saved filter for audio processing"""
        
        if filter_name in st.session_state.get('designed_filters', {}):
            filter_data = st.session_state.designed_filters[filter_name]
            st.session_state.current_b = filter_data['b']
            st.session_state.current_a = filter_data['a']
            st.session_state.current_filter = filter_data
            st.session_state.sampling_freq = filter_data['fs']
            return True
        return False

    def _render_learn_more(self):
        """Render educational content about audio processing and filtering"""
        
        with st.expander("📚 Learn More About Audio Processing and Filtering"):
            st.markdown("""
            <div class="learn-more">
                <h3 style='color: #FFC1CC;'>Understanding Audio Processing and Digital Filters</h3>
                
                **Audio Processing**:
                - **What is it?** Audio processing involves manipulating audio signals to achieve desired effects, such as filtering, amplification, or noise reduction.
                - **In this tool**: You upload an audio file, apply a digital filter (defined by coefficients b and a), and adjust gain to modify the output.
                
                **Digital Filters**:
                - **What are they?** Digital filters process audio by combining input samples (x[n]) and past outputs (y[n]) using coefficients:
                  ```
                  y[n] = b₀x[n] + b₁x[n-1] + ... - a₁y[n-1] - a₂y[n-2] - ...
                  ```
                  - Numerator coefficients (b) shape the input contribution.
                  - Denominator coefficients (a) introduce feedback (for IIR filters).
                - **Role**: Filters can remove unwanted frequencies (e.g., low-pass removes high frequencies) or enhance specific bands.
                
                **Poles and Zeros**:
                - **Zeros**: Roots of the numerator polynomial (b coefficients). They create nulls in the frequency response, attenuating specific frequencies.
                - **Poles**: Roots of the denominator polynomial (a coefficients). They cause resonance and determine stability. A filter is stable if all poles are inside the unit circle (|z| < 1).
                - **Impact on Audio**: Zeros reduce certain frequencies (e.g., a zero at high frequencies for a low-pass filter), while poles amplify nearby frequencies. Unstable poles can cause distortion or oscillation.
                
                **Waveform Visualization**:
                - Shows the amplitude of the audio signal over time. Compare original and filtered waveforms to see the filter’s effect.
                
                **Spectrogram Visualization**:
                - Displays frequency content over time. Brighter colors indicate higher amplitude at specific frequencies.
                - Compare original and filtered spectrograms to observe how the filter alters frequency components (e.g., removing high frequencies in a low-pass filter).
                
                **Key Metrics**:
                - **RMS Level**: Measures the average loudness of the audio.
                - **Peak Amplitude**: Indicates the maximum signal level. Values near 1.0 risk clipping (distortion).
                - **Clipping Risk**: Highlighted in red if peak amplitude exceeds 0.95, warning of potential distortion.
                
                **Why It Matters**:
                - Filters are used in audio engineering to equalize sound, remove noise, or create effects.
                - Understanding poles and zeros helps predict how a filter will affect audio frequencies.
                
                **Try It Out**:
                - Upload different audio files (e.g., music, speech) and apply various filters.
                - Adjust gain and observe clipping risks.
                - Use the zoom level to inspect waveform details.
                - Check the spectrogram to see frequency changes.
                - Explore pole-zero impact to understand filter behavior.
            </div>
            """, unsafe_allow_html=True)
