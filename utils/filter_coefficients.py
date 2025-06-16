import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from scipy import signal
from fractions import Fraction

class FilterCoefficientsTab:
    """
    Filter Coefficients Section - Analysis and manipulation of filter coefficients with poles and zeros visualization
    """

    def __init__(self):
        self.display_formats = {
            'decimal': 'Decimal',
            'scientific': 'Scientific',
            'fraction': 'Fraction'
        }

    def render(self):
        """Main render function for the filter coefficients section"""

        # Custom CSS with baby pink color scheme
        st.markdown("""
        <style>
            .coeff-panel {
                background: linear-gradient(135deg, #FFC1CC 0%, #FFABB6 100%);
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .coeff-panel h3 {
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

        st.markdown('<div class="section-header">📋 Filter Coefficients</div>', unsafe_allow_html=True)

        # Create main layout
        col1, col2 = st.columns([1, 3])

        with col1:
            self._render_coeff_panel()

        with col2:
            self._render_coeff_visualization()

        # Detailed analysis section
        st.markdown("---")
        self._render_coeff_analysis()

        # Educational section
        self._render_learn_more()

    def _render_coeff_panel(self):
        """Render the coefficient control panel"""

        st.markdown("""
        <div class="coeff-panel">
            <h3>📋 Coefficient Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)

        # Coefficient format selection with tooltip
        st.markdown("""
        <div title="Choose how to display filter coefficients:\n- Decimal: Fixed-point numbers\n- Scientific: Exponential notation\n- Fraction: Rational numbers">
        """, unsafe_allow_html=True)
        coeff_format = st.selectbox(
            "🎯 Coefficient Format",
            options=list(self.display_formats.keys()),
            format_func=lambda x: self.display_formats[x],
            help="Choose how to display coefficients",
            key="coeff_format"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Load saved filter
        if st.session_state.get('designed_filters', {}):
            st.markdown("""
            <div title="Select a previously designed filter to load its coefficients">
            """, unsafe_allow_html=True)
            filter_name = st.selectbox(
                "🗂️ Load Saved Filter",
                options=["Current Filter"] + list(st.session_state.designed_filters.keys()),
                help="Select a saved filter to analyze its coefficients",
                key="coeff_load_filter"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            if filter_name != "Current Filter" and st.button("Load Filter", use_container_width=True, key="load_filter_btn"):
                if self.load_filter(filter_name):
                    st.success(f"✅ Loaded filter '{filter_name}'")
                    st.rerun()

        # Display coefficients
        if st.session_state.get('current_b') is not None and st.session_state.get('current_a') is not None:
            b = st.session_state.current_b
            a = st.session_state.current_a

            # Ensure coefficients are float arrays to handle NaN values
            b = np.asarray(b, dtype=float)
            a = np.asarray(a, dtype=float)

            # Format coefficients based on selection
            if coeff_format == 'scientific':
                b_display = [f"{x:.2e}" for x in b]
                a_display = [f"{x:.2e}" for x in a]
            elif coeff_format == 'fraction':
                b_display = [str(Fraction(x).limit_denominator(1000)) for x in b]
                a_display = [str(Fraction(x).limit_denominator(1000)) for x in a]
            else:  # decimal
                b_display = [f"{x:.6f}" for x in b]
                a_display = [f"{x:.6f}" for x in a]

            # Create dataframe for display - handle different lengths properly
            max_len = max(len(b_display), len(a_display))
            b_display_padded = b_display + [''] * (max_len - len(b_display))
            a_display_padded = a_display + [''] * (max_len - len(a_display))

            st.markdown("""
            <div title="Coefficients define the filter's difference equation:\ny[n] = Σ(b_k * x[n-k]) - Σ(a_k * y[n-k])">
            """, unsafe_allow_html=True)
            df = pd.DataFrame({
                'Numerator (b)': b_display_padded,
                'Denominator (a)': a_display_padded
            })
            st.dataframe(df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Export options
            st.markdown("### 💾 Export Coefficients")
            col_export1, col_export2 = st.columns(2)
            with col_export1:
                # --- Robust Export Section ---
                max_len = max(len(b_display), len(a_display))
                b_export_padded = b_display + [''] * (max_len - len(b_display))
                a_export_padded = a_display + [''] * (max_len - len(a_display))
                export_df = pd.DataFrame({
                    'Numerator (b)': b_export_padded,
                    'Denominator (a)': a_export_padded
                })
                csv = export_df.to_csv(index=True)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="filter_coefficients.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="coeff_download_csv"
                )
            with col_export2:
                json_data = {
                    'numerator_b': b.tolist(),
                    'denominator_a': a.tolist()
                }
                st.download_button(
                    label="📥 Download as JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name="filter_coefficients.json",
                    mime="application/json",
                    use_container_width=True,
                    key="coeff_download_json"
                )

            # Import coefficients
            st.markdown("### 📤 Import Coefficients")
            uploaded_file = st.file_uploader("Upload CSV or JSON file", type=['csv', 'json'], key="coeff_upload")
            if uploaded_file and st.button("Import Coefficients", use_container_width=True, key="import_coeff_btn"):
                self._import_coefficients(uploaded_file)

            # Coefficient editor
            with st.expander("✏️ Edit Coefficients"):
                st.markdown("### ✏️ Coefficient Editor")
                st.markdown("""
                <div title="Enter comma-separated values for numerator (b) and denominator (a) coefficients">
                """, unsafe_allow_html=True)
                b_new = st.text_area("Numerator Coefficients (b) (comma-separated)", 
                                    value=','.join(map(str, b)), 
                                    help="Enter new numerator coefficients", 
                                    key="edit_b")
                a_new = st.text_area("Denominator Coefficients (a) (comma-separated)", 
                                    value=','.join(map(str, a)), 
                                    help="Enter new denominator coefficients", 
                                    key="edit_a")
                st.markdown("</div>", unsafe_allow_html=True)
                if st.button("Apply Changes", use_container_width=True, key="apply_coeff_changes"):
                    try:
                        b_new = np.array([float(x.strip()) for x in b_new.split(',') if x.strip()])
                        a_new = np.array([float(x.strip()) for x in a_new.split(',') if x.strip()])
                        if len(b_new) == 0 or len(a_new) == 0:
                            raise ValueError("Coefficient arrays cannot be empty")
                        st.session_state.current_b = b_new
                        st.session_state.current_a = a_new
                        st.session_state.current_filter = {
                            'method': 'custom',
                            'type': 'custom',
                            'order': max(len(b_new), len(a_new)) - 1,
                            'frequencies': [],
                            'fs': st.session_state.sampling_freq,
                            'b': b_new,
                            'a': a_new
                        }
                        st.success("✅ Coefficients updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error updating coefficients: {str(e)}")

    def _render_coeff_visualization(self):
        """Render the coefficient and pole-zero visualization panel"""

        st.markdown("""
        <div class="viz-card">
            <h3 style='color: #FFC1CC; text-align: center;'>📊 Coefficient and Pole-Zero Visualization</h3>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get('current_b') is not None and st.session_state.get('current_a') is not None:
            b = np.asarray(st.session_state.current_b, dtype=float)
            a = np.asarray(st.session_state.current_a, dtype=float)

            # Visualization options
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                show_unit_circle = st.checkbox("Show Unit Circle", value=True, key="show_unit_circle")
            with col_viz2:
                show_grid = st.checkbox("Show Grid", value=True, key="show_grid")

            # Create subplots: coefficients and pole-zero plot
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Numerator Coefficients (b)', 'Denominator Coefficients (a)', 'Pole-Zero Plot'),
                vertical_spacing=0.1,
                specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "scatter"}]]
            )

            # Numerator coefficients (b)
            fig.add_trace(
                go.Bar(
                    x=np.arange(len(b)),
                    y=b,
                    name='Numerator (b)',
                    marker=dict(color='#FFC1CC'),
                    hovertemplate='Index: %{x}<br>Value: %{y:.6f}<br>Numerator coefficient b[%{x}]<extra></extra>'
                ),
                row=1, col=1
            )

            # Denominator coefficients (a)
            fig.add_trace(
                go.Bar(
                    x=np.arange(len(a)),
                    y=a,
                    name='Denominator (a)',
                    marker=dict(color='#FFB6C1'),
                    hovertemplate='Index: %{x}<br>Value: %{y:.6f}<br>Denominator coefficient a[%{x}]<extra></extra>'
                ),
                row=2, col=1
            )

            # Pole-zero plot
            zeros = np.roots(b) if len(b) > 1 else np.array([])
            poles = np.roots(a) if len(a) > 1 else np.array([])

            # Unit circle
            if show_unit_circle:
                theta = np.linspace(0, 2*np.pi, 100)
                x_circle = np.cos(theta)
                y_circle = np.sin(theta)
                fig.add_trace(
                    go.Scatter(
                        x=x_circle, y=y_circle, mode='lines', name='Unit Circle',
                        line=dict(color='#FFABB6', dash='dash'),
                        hovertemplate='Unit Circle: |z| = 1<extra></extra>'
                    ),
                    row=3, col=1
                )

            # Zeros
            if len(zeros) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=zeros.real, y=zeros.imag, mode='markers', name='Zeros',
                        marker=dict(symbol='circle', size=10, color='#FFC1CC', line=dict(color='#333', width=1)),
                        hovertemplate='Zero: (%{x:.3f}, %{y:.3f}j)<br>Magnitude: %{customdata:.3f}<extra></extra>',
                        customdata=np.abs(zeros)
                    ),
                    row=3, col=1
                )

            # Poles
            if len(poles) > 0:
                # Color-code unstable poles
                pole_mags = np.abs(poles)
                pole_colors = ['#FFB6C1' if mag < 1 else '#FF0000' for mag in pole_mags]
                fig.add_trace(
                    go.Scatter(
                        x=poles.real, y=poles.imag, mode='markers', name='Poles',
                        marker=dict(symbol='x', size=10, color=pole_colors, line=dict(color='#333', width=1)),
                        hovertemplate='Pole: (%{x:.3f}, %{y:.3f}j)<br>Magnitude: %{customdata:.3f}<extra></extra>',
                        customdata=pole_mags
                    ),
                    row=3, col=1
                )

            # Update layout
            fig.update_layout(
                height=900,
                template="plotly_white",
                showlegend=True,
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )

            # Update axes
            fig.update_xaxes(title_text="Coefficient Index", row=1, col=1, showgrid=show_grid)
            fig.update_xaxes(title_text="Coefficient Index", row=2, col=1, showgrid=show_grid)
            fig.update_xaxes(title_text="Real Part", row=3, col=1, range=[-1.5, 1.5], showgrid=show_grid)
            fig.update_yaxes(title_text="Value", row=1, col=1, showgrid=show_grid)
            fig.update_yaxes(title_text="Value", row=2, col=1, showgrid=show_grid)
            fig.update_yaxes(title_text="Imaginary Part", row=3, col=1, range=[-1.5, 1.5], scaleanchor="x", scaleratio=1, showgrid=show_grid)

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
            <div class="viz-card" style='text-align: center; padding: 2rem;'>
                <h3 style='color: #FFC1CC;'>🎯 No Coefficients Available</h3>
                <p style='color: #555; font-size: 1rem;'>
                    Design a filter in the Filter Design section or load a saved filter to see coefficient and pole-zero visualizations
                </p>
            </div>
            """, unsafe_allow_html=True)

    def _render_coeff_analysis(self):
        """Render detailed coefficient analysis and metrics"""

        st.markdown('<div class="section-header">🔍 Coefficient Analysis</div>', unsafe_allow_html=True)

        if st.session_state.get('current_b') is not None and st.session_state.get('current_a') is not None:
            b = np.asarray(st.session_state.current_b, dtype=float)
            a = np.asarray(st.session_state.current_a, dtype=float)

            # Calculate metrics
            b_mean = np.mean(np.abs(b)) if len(b) > 0 else 0.0
            b_max = np.max(np.abs(b)) if len(b) > 0 else 0.0
            a_mean = np.mean(np.abs(a)) if len(a) > 0 else 0.0
            a_max = np.max(np.abs(a)) if len(a) > 0 else 0.0

            # Create three columns for metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 📊 Numerator (b) Metrics")
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📏 Mean Absolute Value</h4>
                    <p style='font-size: 1.2rem;'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Average of absolute coefficients</p>
                </div>
                """.format(b_mean), unsafe_allow_html=True)
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>🔺 Max Absolute Value</h4>
                    <p style='font-size: 1.2rem;'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Largest absolute coefficient</p>
                </div>
                """.format(b_max), unsafe_allow_html=True)

            with col2:
                st.markdown("#### 📊 Denominator (a) Metrics")
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📏 Mean Absolute Value</h4>
                    <p style='font-size: 1.2rem;'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Average of absolute coefficients</p>
                </div>
                """.format(a_mean), unsafe_allow_html=True)
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>🔺 Max Absolute Value</h4>
                    <p style='font-size: 1.2rem;'>{:.6f}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Largest absolute coefficient</p>
                </div>
                """.format(a_max), unsafe_allow_html=True)

            with col3:
                st.markdown("#### ⚙️ Coefficient Properties")
                num_coeffs = len(b)
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📈 Number of Coefficients (b)</h4>
                    <p style='font-size: 1.2rem;'>{}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Numerator coefficient count</p>
                </div>
                """.format(num_coeffs), unsafe_allow_html=True)
                den_coeffs = len(a)
                st.markdown("""
                <div class="metric-card">
                    <h4 style='color: #FFC1CC;'>📈 Number of Coefficients (a)</h4>
                    <p style='font-size: 1.2rem;'>{}</p>
                    <p style='color: #555; font-size: 0.9rem;'>Denominator coefficient count</p>
                </div>
                """.format(den_coeffs), unsafe_allow_html=True)

            # Stability and pole-zero counts
            poles = np.roots(a) if len(a) > 1 else np.array([])
            zeros = np.roots(b) if len(b) > 1 else np.array([])
            is_stable = np.all(np.abs(poles) < 1) if len(poles) > 0 else True
            stability_text = "✅ Stable" if is_stable else "⚠️ Unstable"
            st.markdown("""
            <div class="metric-card">
                <h4 style='color: #FFC1CC;'>🔧 Filter Stability</h4>
                <p style='font-size: 1.2rem;'>{}</p>
                <p style='color: #555; font-size: 0.9rem;'>Based on pole locations</p>
            </div>
            """.format(stability_text), unsafe_allow_html=True)

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

        else:
            st.info("Please design a filter in the 'Filter Design' section or load a saved filter to analyze coefficients.")

    def _import_coefficients(self, uploaded_file):
        """Import coefficients from uploaded CSV or JSON file"""

        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'Numerator (b)' not in df.columns or 'Denominator (a)' not in df.columns:
                    raise ValueError("CSV must contain 'Numerator (b)' and 'Denominator (a)' columns")
                b = df['Numerator (b)'].dropna().astype(float).to_numpy()
                a = df['Denominator (a)'].dropna().astype(float).to_numpy()
            elif uploaded_file.name.endswith('.json'):
                json_data = json.load(uploaded_file)
                if 'numerator_b' not in json_data or 'denominator_a' not in json_data:
                    raise ValueError("JSON must contain 'numerator_b' and 'denominator_a' keys")
                b = np.array(json_data['numerator_b'], dtype=float)
                a = np.array(json_data['denominator_a'], dtype=float)
            else:
                st.error("❌ Unsupported file format. Please upload CSV or JSON.")
                return

            if len(b) == 0 or len(a) == 0:
                raise ValueError("Coefficient arrays cannot be empty")

            st.session_state.current_b = b
            st.session_state.current_a = a
            st.session_state.current_filter = {
                'method': 'imported',
                'type': 'custom',
                'order': max(len(b), len(a)) - 1,
                'frequencies': [],
                'fs': st.session_state.sampling_freq,
                'b': b,
                'a': a
            }
            st.success("✅ Coefficients imported successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error importing coefficients: {str(e)}")

    def load_filter(self, filter_name):
        """Load a saved filter's coefficients"""

        if filter_name in st.session_state.get('designed_filters', {}):
            filter_data = st.session_state.designed_filters[filter_name]
            st.session_state.current_b = filter_data['b']
            st.session_state.current_a = filter_data['a']
            st.session_state.current_filter = filter_data
            st.session_state.sampling_freq = filter_data['fs']
            return True
        return False

    def _render_learn_more(self):
        """Render educational content about coefficients, poles, and zeros"""

        with st.expander("📚 Learn More About Coefficients, Poles, and Zeros"):
            st.markdown("""
            <div class="learn-more">
                <h3 style='color: #FFC1CC;'>Understanding Filter Coefficients, Poles, and Zeros</h3>
                **Filter Coefficients (b, a)**:
                - **What are they?** Coefficients are the numerical weights in a digital filter's difference equation. Numerator coefficients (b) apply to the input signal, while denominator coefficients (a) apply to past outputs (for IIR filters).
                - **Role**: They define how the filter processes the input to produce the output. For example, in the equation:
                  ```
                  y[n] = b₀x[n] + b₁x[n-1] + ... - a₁y[n-1] - a₂y[n-2] - ...
                  ```
                  bₖ and aₖ are the coefficients.
                - **Visualization**: The bar plots above show the values of b and a at each index.
                **Zeros**:
                - **What are they?** Zeros are the values of z where the filter's transfer function H(z) equals zero. They are the roots of the numerator polynomial (formed by b coefficients).
                - **Role**: Zeros create nulls in the frequency response, attenuating specific frequencies. For example, a zero at z = 1 attenuates DC (0 Hz).
                - **Visualization**: Shown as circles (●) in the pole-zero plot. Hover to see their coordinates and magnitude.
                **Poles**:
                - **What are they?** Poles are the values of z where H(z) becomes infinite. They are the roots of the denominator polynomial (formed by a coefficients).
                - **Role**: Poles shape the filter's resonance and determine stability. A filter is stable if all poles are inside the unit circle (|z| < 1). Unstable poles are shown in red.
                - **Visualization**: Shown as x's (✗) in the pole-zero plot. Hover to see their coordinates and magnitude.
                **Unit Circle**:
                - The unit circle (|z| = 1) in the z-plane represents the frequency axis. Points on the circle correspond to frequencies from 0 to the sampling frequency.
                - Poles/zeros near the unit circle strongly affect the frequency response.
                **Why It Matters**:
                - Coefficients are designed to achieve desired filtering (e.g., low-pass, high-pass).
                - Poles and zeros reveal the filter's frequency behavior and stability, helping engineers optimize designs.
                **Try It Out**:
                - Edit coefficients and observe how the pole-zero plot changes.
                - Load different filters to see how poles/zeros affect stability and response.
                - Toggle the unit circle and grid to focus on specific aspects of the plot.
            </div>
            """, unsafe_allow_html=True)
