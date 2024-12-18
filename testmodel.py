from ultralytics import YOLO


dbdict = {
    "1":{
        "roll":"1",
        "name":"parthbhosale",
        "prn":"22010069",
        "branch":"Computer Engineering"
    },
    "2":{
        "roll":"2",
        "name":"saibhosale",
        "prn":"22010078",
        "branch":"Entc"
    }
}


model = YOLO("best.pt")
results = model.predict("parth23.jpg")
result = results[0]
box = result.boxes
class_id = box.cls.item()
class_id = result.names[box.cls[0].item()]
# print(class_id)
print(dbdict.get(class_id))