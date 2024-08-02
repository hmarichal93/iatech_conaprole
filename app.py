
class Pipeline:
    def __init__(self):
        pass

    def load(self, image_path):
        pass

    def yolov5(self, image):
        pass

    def classifier(self):
        pass

    def compute_metrics(self):
        pass

    def print_metrics(self, res):
        pass

    def main(self, image_path):
        image = self.load(image_path)
        res = self.yolov5(image)
        res = self.classifier()
        res = self.compute_metrics()
        self.print_metrics(res)
        return res


def main()