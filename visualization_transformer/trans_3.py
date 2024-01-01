import tkinter as tk
from tkinter import Canvas
import time

class TransformerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Transformer Model Visualization")

        self.canvas = Canvas(root, width=800, height=500)
        self.canvas.pack()

        # Draw the input layer
        self.input_layer = self.canvas.create_rectangle(50, 400, 750, 450, fill="lightgrey", tags="input_layer")
        self.canvas.create_text(400, 425, text="Input Layer", font=("Arial", 10))

        # Draw encoder and decoder blocks
        self.encoder = self.canvas.create_rectangle(50, 150, 250, 350, fill="skyblue", tags="encoder")
        self.decoder = self.canvas.create_rectangle(550, 150, 750, 350, fill="lightgreen", tags="decoder")

        # Labels for encoder and decoder
        self.canvas.create_text(150, 130, text="Encoder")
        self.canvas.create_text(650, 130, text="Decoder")

        # Details in the encoder
        self.canvas.create_text(150, 170, text="Multiple Layers", font=("Arial", 10))
        for i in range(3):
            y_start = 200 + i * 50
            y_end = y_start + 40
            self.canvas.create_rectangle(70, y_start, 230, y_end, outline="black", fill="white")
            self.canvas.create_text(150, y_start + 20, text=f"Layer {i+1}", font=("Arial", 8))

        # Data flow elements
        self.data_flow = []

    def animate_data_flow(self):
        # Simulate data flow from input layer to encoder and then to decoder
        for i in range(5):
            input_x = 100 + i * 120
            input_y = 430
            data = self.canvas.create_oval(input_x, input_y, input_x + 20, input_y + 20, fill="red", tags="data")
            self.data_flow.append(data)

            # Move up to the encoder
            for _ in range(30):
                self.canvas.move(data, 0, -5)
                self.root.update()
                time.sleep(0.05)

            # Move through the encoder
            for _ in range(20):
                self.canvas.move(data, 2, 0)
                self.root.update()
                time.sleep(0.05)

            # Move to the decoder
            for _ in range(100):
                self.canvas.move(data, 3, 0)
                self.root.update()
                time.sleep(0.05)

            # Clean up
            self.canvas.delete(data)

    def start(self):
        self.animate_data_flow()
        self.root.mainloop()

# Create and run the application
root = tk.Tk()
app = TransformerGUI(root)
app.start()
