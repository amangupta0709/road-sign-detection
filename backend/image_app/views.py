from rest_framework import serializers
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import base64
from io import BytesIO
from PIL import Image
import subprocess
import glob


class ImageSerializer(serializers.Serializer):
    img = serializers.FileField()
    img_type = serializers.ChoiceField(choices=["IMG1", "IMG2"])


class ImageAPIView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        serializer = ImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        img_file = serializer.validated_data["img"]
        img_type = serializer.validated_data["img_type"]

        processed_img_data = self.process_image(img_file, img_type)

        return Response({"processed_img": processed_img_data})

    def process_image(self, img_file, img_type):
        file_path = "Learning-to-See-in-the-Dark/dataset/Sony/123456789.dng"

        if img_type == "IMG1":
            with open(file_path, "wb") as f:
                f.write(img_file.read())

            subprocess.run(
                "venv/bin/python test_Sony.py",
                shell=True,
                executable="/bin/zsh",
                cwd="Learning-to-See-in-the-Dark/",
            )

            result_img_path = "Learning-to-See-in-the-Dark/result_Sony/final/1_out.png"

        elif img_type == "IMG2":
            subprocess.run(
                "../venv/bin/python3 detect.py --weights best.pt --source ../Learning-to-See-in-the-Dark/result_Sony/final/1_out.png",
                shell=True,
                executable="/bin/zsh",
                cwd="yolov5/",
            )
            result_dir = max(glob.glob("yolov5/runs/detect/*"))
            result_img_path = f"{result_dir}/1_out.png"

        with Image.open(result_img_path) as img:
            img = img.resize((300, 300))
            buffered = BytesIO()

            img.save(buffered, format="JPEG")
            img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_data
