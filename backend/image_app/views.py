from rest_framework import serializers
from rest_framework.views import APIView
from rest_framework.response import Response
import base64
from io import BytesIO
from PIL import Image


class ImageSerializer(serializers.Serializer):
    img = serializers.CharField()
    img_type = serializers.ChoiceField(choices=["IMG1", "IMG2"])


class ImageAPIView(APIView):
    def post(self, request):
        serializer = ImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        img_data = serializer.validated_data["img"]
        img_type = serializer.validated_data["img_type"]

        img_bytes = base64.b64decode(img_data)
        processed_img = self.process_image(img_bytes, img_type)

        buffered = BytesIO()
        if processed_img.mode in ["RGBA", "P"]:
            processed_img = processed_img.convert("RGB")
        processed_img.save(buffered, format="JPEG")
        processed_img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return Response({"processed_img": processed_img_data})

    def process_image(self, img_bytes, img_type):
        img = Image.open(BytesIO(img_bytes))

        return img
