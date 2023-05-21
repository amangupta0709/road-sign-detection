from django.urls import path
from image_app.views import ImageAPIView

urlpatterns = [path("process-image/", ImageAPIView.as_view(), name="process-image")]
