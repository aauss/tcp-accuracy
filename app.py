import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("aauss/tcp_accuracy")
launch_gradio_widget(module)
