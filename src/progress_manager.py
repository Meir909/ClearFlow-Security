import streamlit as st
import time
import threading

class ProgressManager:
    
    @staticmethod
    def create_animated_progress_bar(label, max_value=100):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        return progress_text, progress_bar
    
    @staticmethod
    def update_progress(progress_text, progress_bar, current, total, label):
        percentage = int((current / total) * 100) if total > 0 else 0
        progress_bar.progress(percentage)
        progress_text.markdown(f"**{label}**: {current}/{total} ({percentage}%)")
    
    @staticmethod
    def create_loading_skeleton(height=200):
        st.markdown(f"""
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; height: {height}px; background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                    background-size: 200% 100%; animation: loading 1.5s infinite;">
            <div style="height: 20px; background: #ddd; margin-bottom: 10px; border-radius: 4px;"></div>
            <div style="height: 15px; background: #eee; margin-bottom: 15px; border-radius: 4px; width: 80%;"></div>
            <div style="height: 15px; background: #eee; margin-bottom: 15px; border-radius: 4px; width: 60%;"></div>
            <div style="height: 15px; background: #eee; margin-bottom: 15px; border-radius: 4px; width: 90%;"></div>
        </div>
        <style>
        @keyframes loading {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_success_animation(message):
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #38ef7d, #11998e);">
            <h2 style="color: white; margin: 0;">🎉</h2>
            <p style="color: white; font-weight: bold; margin: 10px 0 0 0;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
    
    @staticmethod
    def show_confetti_animation():
        st.markdown("""
        <div style="position: relative; height: 100px; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 10%; animation: fall 3s linear infinite; font-size: 24px;">🎊</div>
            <div style="position: absolute; top: 0; left: 30%; animation: fall 2.5s linear infinite 0.5s; font-size: 24px;">🎉</div>
            <div style="position: absolute; top: 0; left: 50%; animation: fall 3.5s linear infinite 1s; font-size: 24px;">✨</div>
            <div style="position: absolute; top: 0; left: 70%; animation: fall 2s linear infinite 1.5s; font-size: 24px;">🎈</div>
            <div style="position: absolute; top: 0; left: 90%; animation: fall 4s linear infinite 2s; font-size: 24px;">🎁</div>
        </div>
        <style>
        @keyframes fall {
            0% { transform: translateY(-50px) rotate(0deg); opacity: 1; }
            100% { transform: translateY(150px) rotate(360deg); opacity: 0; }
        }
        </style>
        """, unsafe_allow_html=True)

progress_manager = ProgressManager()