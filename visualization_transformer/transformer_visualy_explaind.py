import tkinter as tk
from tkinter import Canvas
import time

class TransformerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Transformer Visualization")

        self.canvas = Canvas(root, width=600, height=400)
        self.canvas.pack()

        # Draw encoder and decoder blocks
        self.encoder = self.canvas.create_rectangle(50, 50, 150, 350, fill="skyblue", tags="encoder")
        self.decoder = self.canvas.create_rectangle(450, 50, 550, 350, fill="lightgreen", tags="decoder")

        # Labels
        self.canvas.create_text(100, 30, text="Encoder")
        self.canvas.create_text(500, 30, text="Decoder")

        # Data flow elements
        self.data_flow = []

    def animate_data_flow(self):
        # Simulate data entering the encoder
        data = self.canvas.create_oval(40, 170, 60, 190, fill="red", tags="data")
        self.data_flow.append(data)

        for _ in range(50):
            self.canvas.move("data", 2, 0)
            self.root.update()
            time.sleep(0.05)

        # Simulate data transfer from encoder to decoder
        for _ in range(80):
            self.canvas.move("data", 5, 0)
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
