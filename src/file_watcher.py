import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("interactions.csv"):
            print("interactions.csv modifié. Exécution du modèle de recommandations...")
            subprocess.run(["python", "recommend.py"])

if __name__ == "__main__":
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='../../web/data/', recursive=False)
    observer.start()
    print("Surveillance de interactions.csv commencée...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
