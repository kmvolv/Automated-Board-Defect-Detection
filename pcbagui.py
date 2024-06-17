import tkinter as tk
import customtkinter as ctk
from tkdial import ScrollKnob
from PIL import ImageTk, Image
import cv2
import os

from ultralytics import YOLO

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# WINDOW_WIDTH = 1200
# WINDOW_HEIGHT = 800

MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600


class MyFrame(ctk.CTkFrame):
    def __init__(self, master, defect_quant_dict, defect_color_mapping, **kwargs):
        super().__init__(master, **kwargs)

        self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=1)

        header_frame = ctk.CTkFrame(master=self, fg_color="#4c4c4c")
        header_frame.columnconfigure(0, weight=1)
        header_frame.columnconfigure(1, weight=1)

        header1 = ctk.CTkLabel(master=header_frame, text="Defect Classes", font=("Roboto", 25), justify="center", text_color="white")
        header2 = ctk.CTkLabel(master=header_frame, text="Quantity", font=("Roboto", 25), justify="center", text_color="white")

        header1.grid(row=0, column=0, ipadx=10, padx=10, pady=30, sticky="w")
        header2.grid(row=0, column=1, ipadx=10, padx=10, pady=30, sticky="e")

        header_frame.grid(row=0, column=0, padx=10, pady=20, columnspan=2, sticky="nwe")

        # defect_quant_dict = {"missing_hole" : 5, "mouse_bite" : 3, "open_circuit" : 2, "short" : 1}

        row = 1
        for defect, quantity in defect_quant_dict.items():
            # print('This is the color in MyClass : ', defect_color_mapping[defect])
            container_frame = ctk.CTkFrame(master=self, fg_color=self.bgr2hex(defect_color_mapping[defect]))
            container_frame.columnconfigure(0, weight=1)

            d1 = ctk.CTkLabel(master=container_frame, text=defect, font=('Helvetica', 20, 'bold'), justify="center")
            d2 = ctk.CTkLabel(master=container_frame, text=quantity, font=('Helvetica', 20, 'bold'), justify="center")

            d1.grid(row=0, column=0, padx=10, pady=10, sticky="w")
            d2.grid(row=0, column=1, padx=10, pady=10, sticky="e")

            container_frame.grid(row=row, column=0, padx=10, pady=10, columnspan=2, sticky="nwe")           
            row+=1

        self.grid(row=2, column=1, padx=10, pady=20, sticky="new")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.grid_rowconfigure(6, weight=1)
        self.grid_rowconfigure(7, weight=1)

    def bgr2hex(self, bgr):
        return "#{:02x}{:02x}{:02x}".format(bgr[2],bgr[1],bgr[0])


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.model_weights = YOLO("train_yolov8/train/weights/best.pt")
        self.class_mapping = {0: "missing_hole", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: "spur", 5: "spurious_copper"}
        self.color_mapping = {"missing_hole" : (170, 175, 65), "mouse_bite" : (180, 110, 70), "open_circuit" : (225, 160, 0), "short" : (50, 165, 230), "spur" : (44, 100, 215), "spurious_copper" : (145, 75, 175)}
        self.input_img = None
        self.orig_img = None
        self.img_w = None
        self.img_h = None
        self.cls_thresh = 0.5

        self.defect_quants = None
        self.threshold_knob = None

        self.title("PCBA Defect Detection")

        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight}")
        self.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        
        self.inference_frame = ctk.CTkFrame(master=self)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=4)
        self.rowconfigure(1, weight=1)

        self.inference_frame.grid(row=0, column=0, sticky="nsew")

        self.inference_frame.columnconfigure(0, weight=1)
        self.inference_frame.columnconfigure(1, weight=1)
        self.inference_frame.columnconfigure(2, weight=1)
        self.inference_frame.rowconfigure(0, weight=1)

        self.control_frame = ctk.CTkFrame(master=self)
        self.control_frame.grid(row=1, column=0, sticky="nsew")

        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.rowconfigure(0, weight=1)

        # self.columnconfigure(0, weight=1)
        # self.columnconfigure(1, weight=1, minsize=75)
        # self.columnconfigure(2, weight=1)

        # self.rowconfigure(0, weight=4, minsize=100)
        # self.rowconfigure(1, weight=1, minsize=100)
        

        self.btn_frame = ctk.CTkFrame(master = self.control_frame, fg_color="#cee9ff")
        self.btn_frame.columnconfigure(0, weight=1)
        self.btn_frame.columnconfigure(1, weight=1)
        # self.btn_frame.columnconfigure(2, weight=1)
        # self.btn_frame.columnconfigure(3, weight=1)
        self.btn_frame.rowconfigure(0, weight=1)

        self.img_frame = ctk.CTkFrame(master = self.inference_frame)
        self.img_frame.columnconfigure(1, weight=1)

        # self.defect_quants.grid(row=0, column=0, sticky="nsew")
        self.img_frame.grid(row=0, column=1, sticky="nsew")
        # self.defect_confs.grid(row=0, column=2, sticky="nsew")
                

        self.btn_frame.grid(row=1, column=0, sticky="nsew")

        self.set_preview_pic("./GUIpics/default_img.png")

        # upload_label = ctk.CTkLabel(master=self.btn_frame, text="Upload Image", text_color="black", bg_color="yellow")
        upload_btn = ctk.CTkButton(master=self.btn_frame, text="Upload Image", command=self.select_pic)
        # detect_label = ctk.CTkLabel(master=self.btn_frame, text="Detect Defects", text_color="black", bg_color="orange")
        detect_btn = ctk.CTkButton(master=self.btn_frame, text="Detect Defects", command=self.start_inference)

        # upload_label.grid(row=0, column=0, padx=10, pady=20, sticky="nws", ipadx=20, ipady=20)
        upload_btn.grid(row=0, column=0, padx=10, pady=20, sticky="nws", ipadx=20, ipady=20)
        # detect_label.grid(row=0, column=4, padx=10, pady=20, sticky="nes", ipadx=20, ipady=20)
        detect_btn.grid(row=0, column=3, padx=10, pady=20, sticky="nes", ipadx=20, ipady=20)

    def set_preview_pic(self, filepath, new_img=None):
        global img
        if new_img is None:
            img = Image.open(filepath)
            img = img.resize((860,860), Image.LANCZOS)
            try:
                self.defect_quants.grid_forget()
                self.defect_quants.destroy()
            except:
                pass
        else:
            img = new_img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        self.img_w, self.img_h = img.size
        # img = img.resize((self.winfo_screenwidth()//4,self.winfo_screenheight()//4), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        
        display_pic = tk.Label(self.img_frame, image=img)
        # display_pic.place(relx=0.5, rely=0.5, anchor="center")
        display_pic.grid(row=1, column=1, padx=10, pady=(20,0))
    
    def select_pic(self):
        global filename
        filename = tk.filedialog.askopenfilename(
            initialdir = os.getcwd(),
            title = "Select Board Image",
            filetypes = (("jpg images", "*.jpg"), ("png images", "*.png"), ("jpeg images", "*.jpeg"))
        )

        try:
            self.set_preview_pic(filename)
            self.input_img = filename
            self.orig_img = cv2.imread(filename)
            self.orig_img = cv2.resize(self.orig_img, (860, 860))
        except:
            self.set_preview_pic("./GUIpics/default_img.png")
            self.input_img = None

    def start_inference(self):
        if self.input_img is not None:
            self.model_predict()
        else:
            #TODO Give an alert
            pass

    def scale_coordinates(self, original_coords, original_resolution, new_resolution):
        # original_coords: (left, top, right, bottom)
        # original_resolution: (original_width, original_height)
        # new_resolution: (new_width, new_height)
        
        # Calculate scaling factors for width and height
        width_scale = new_resolution[0] / original_resolution[0]
        height_scale = new_resolution[1] / original_resolution[1]

        # Scale the coordinates
        scaled_left = int(original_coords[0] * width_scale)
        scaled_top = int(original_coords[1] * height_scale)
        scaled_right = int(original_coords[2] * width_scale)
        scaled_bottom = int(original_coords[3] * height_scale)

        return (scaled_left, scaled_top, scaled_right, scaled_bottom)
    
    def new_thresh_update(self, new_thresh, results):
        infer_img = self.orig_img.copy()

        defect_quant_dict = {}

        for i in range(len(results.boxes)):
            # print(f"\n\n =============== The {i}th classification is : {self.class_mapping[results.boxes.cls.tolist()[i]]} ============= \n")

            center_x = results.boxes.xywhn[i][0]
            center_y = results.boxes.xywhn[i][1]
            defect_w = results.boxes.xywhn[i][2]
            defect_h = results.boxes.xywhn[i][3]

            l = int((center_x - defect_w/2)* self.img_w)
            r = int((center_x + defect_w/2)* self.img_w)
            t = int((center_y - defect_h/2)* self.img_h)
            b = int((center_y + defect_h/2)* self.img_h)

            pred_conf = str(results.boxes.conf.tolist()[i])[:4]

            if(float(pred_conf) < new_thresh/100):
                continue

            l, t, r, b = self.scale_coordinates((l,t,r,b), (self.img_w, self.img_h), (860, 860))

            class_label = self.class_mapping[results.boxes.cls.tolist()[i]]
            defect_quant_dict[class_label] = defect_quant_dict.get(class_label, 0) + 1

            # Bounding box over defect
            cv2.rectangle(infer_img, (l,t), (r,b), self.color_mapping[class_label], 2)

            # print("Hi this is the new function, and here is the prediction confidence ; ", pred_conf, "  and ig here's the type of it as well : ", type(pred_conf))

            # Find space required by Label Text
            (lw, lh), _ = cv2.getTextSize(pred_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Label Text
            cv2.rectangle(infer_img, (l, t-20), (l+lw, t), self.color_mapping[class_label], -1)
            cv2.putText(infer_img, pred_conf, (l, t-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        print("This is the final defect quantities : ", defect_quant_dict)
        self.set_preview_pic("", infer_img)
        # self.defect_quants = MyFrame(master=self.inference_frame, fg_color="gray", defect_quant_dict=defect_quant_dict, defect_color_mapping=self.color_mapping)

    def model_predict(self):
        infer_img = cv2.imread(self.input_img)
        [results] = self.model_weights(infer_img, save=False)
        
        # Resizing image
        infer_img = cv2.resize(infer_img, (860, 860))

        defect_quant_dict = {}

        for i in range(len(results.boxes)):
            # print(f"\n\n =============== The {i}th classification is : {self.class_mapping[results.boxes.cls.tolist()[i]]} ============= \n")

            center_x = results.boxes.xywhn[i][0]
            center_y = results.boxes.xywhn[i][1]
            defect_w = results.boxes.xywhn[i][2]
            defect_h = results.boxes.xywhn[i][3]

            l = int((center_x - defect_w/2)* self.img_w)
            r = int((center_x + defect_w/2)* self.img_w)
            t = int((center_y - defect_h/2)* self.img_h)
            b = int((center_y + defect_h/2)* self.img_h)

            l, t, r, b = self.scale_coordinates((l,t,r,b), (self.img_w, self.img_h), (860, 860))

            class_label = self.class_mapping[results.boxes.cls.tolist()[i]]
            defect_quant_dict[class_label] = defect_quant_dict.get(class_label, 0) + 1

            # Bounding box over defect
            cv2.rectangle(infer_img, (l,t), (r,b), self.color_mapping[class_label], 2)

            pred_conf = str(results.boxes.conf.tolist()[i])[:4]

            # Find space required by Label Text
            (lw, lh), _ = cv2.getTextSize(pred_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Label Text
            cv2.rectangle(infer_img, (l, t-20), (l+lw, t), self.color_mapping[class_label], -1)
            cv2.putText(infer_img, pred_conf, (l, t-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # To pop in defect view after inference : 
        self.defect_quants = MyFrame(master=self.inference_frame, fg_color="gray", defect_quant_dict=defect_quant_dict, defect_color_mapping=self.color_mapping)
        # self.defect_confs = MyFrame(master=self.inference_frame, fg_color="red")

        # ============= For Threshold Knob =============
        dummy_frame = ctk.CTkFrame(master=self.defect_quants, corner_radius=0)
        self.threshold_knob = ScrollKnob(dummy_frame, text=" ", start=0, end=100, steps=1, radius=200, bar_color="#242424", 
                   progress_color="yellow", outer_color="yellow", outer_length=10, 
                   border_width=30, start_angle=270, inner_width=0, outer_width=5, text_font="calibri 20", 
                   text_color="white", fg="#212325", command=lambda x : self.new_thresh_update(x, results))
        

        threshold_label = ctk.CTkLabel(master=dummy_frame, text="Threshold", font=("Arial", 18, 'bold'), justify="center")

        dummy_frame.grid(row=7, column=0, columnspan=2, sticky="sew")
        dummy_frame.columnconfigure(0, weight=1)
        dummy_frame.columnconfigure(1, weight=1)
        dummy_frame.rowconfigure(0, weight=1)

        self.threshold_knob.grid(row=0, column=0, padx=(82,0), pady=10, columnspan=2, sticky="nsew")
        threshold_label.grid(row=1, column=0, padx=10, pady=10, columnspan=2, sticky="nsew")
        # ===============================================

        self.defect_quants.columnconfigure(0, weight=1)
        # self.defect_confs.columnconfigure(2, weight=1)

        self.defect_quants.grid(row=0, column=0, sticky="nsew")
        # self.defect_confs.grid(row=0, column=2, sticky="nsew")
        
        print("This is the final defect quantities : ", defect_quant_dict)
        self.set_preview_pic("", infer_img)


if __name__ == "__main__":
    app = App()
    app.mainloop()