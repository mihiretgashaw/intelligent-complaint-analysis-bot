import gradio as gr
from rag_pipeline import generate_answer

def gradio_interface(question):
    return generate_answer(question)

gr.Interface(fn=gradio_interface,
             inputs=gr.Textbox(lines=2, placeholder="Ask a question about customer complaints..."),
             outputs="text",
             title="CrediTrust Complaint Assistant").launch()
