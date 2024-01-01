import tkinter as tk
from tkinter import Canvas
import time

class TransformerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Transformer Visualization")

        self.canvas = Canvas(root, width=800, height=400)
        self.canvas.pack()

        # Draw encoder and decoder blocks
        self.encoder = self.canvas.create_rectangle(50, 50, 250, 350, fill="skyblue", tags="encoder")
        self.decoder = self.canvas.create_rectangle(550, 50, 750, 350, fill="lightgreen", tags="decoder")

        # Labels for encoder and decoder
        self.canvas.create_text(150, 30, text="Encoder")
        self.canvas.create_text(650, 30, text="Decoder")

        # Adding details to the encoder
        self.canvas.create_text(150, 70, text="Multiple Layers", font=("Arial", 10))
        for i in range(3):
            y_start = 100 + i * 80
            y_end = y_start + 60
            self.canvas.create_rectangle(70, y_start, 230, y_end, outline="black", fill="white")
            self.canvas.create_text(150, y_start + 20, text="Self-Attention", font=("Arial", 8))
            self.canvas.create_text(150, y_start + 40, text="Feed-Forward", font=("Arial", 8))

        # Data flow elements
        self.data_flow = []

    def animate_data_flow(self):
        # Simulate data entering the encoder
        data = self.canvas.create_oval(40, 170, 60, 190, fill="red", tags="data")
        self.data_flow.append(data)

        for _ in range(100):
            self.canvas.move("data", 2, 0)
            self.root.update()
            time.sleep(0.05)

        # Simulate data transfer from encoder to decoder
        for _ in range(150):
            self.canvas.move("data", 3, 0)
            self.root.update()
            time.sleep(0.05)

        # Clean up
        self.canvas.delete("data")

    def start(self):
        self.animate_data_flow()
        self.root.mainloop()

# Create and run the application
root = tk.Tk()
app = TransformerGUI(root)
app.start()
