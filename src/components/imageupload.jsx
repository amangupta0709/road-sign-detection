import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import Button from "@mui/material/Button";
import Skeleton from "@mui/material/Skeleton";
import axios from "axios";
import React, { useState } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const ImageUpload = () => {
  const [loadingImage1, setLoadingImage1] = useState(null);
  const [loadingImage2, setLoadingImage2] = useState(null);
  const [image1String, setImage1String] = useState("");
  const [image2String, setImage2String] = useState("");

  const convertBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsDataURL(file);
      fileReader.onload = () => {
        resolve(fileReader.result.split(",")[1]);
      };
      fileReader.onerror = (error) => {
        reject(error);
      };
    });
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("img", file);

    toast.success("Image Uploading", {
      theme: "dark",
      pauseOnHover: false,
    });
    setLoadingImage1(true);
    setLoadingImage2(true);

    // Simulate image upload progress
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Hit API to upload image
    // Replace 'api/upload' with your actual backend API endpoint

    // console.log(file);
    formData.append("img_type", "IMG1");
    axios
      .post("http://127.0.0.1:8001/api/process-image/", formData)
      .then((res) => {
        const imageData = res.data.processed_img;
        // Display the first image
        setLoadingImage1(false);
        setImage1String(imageData);

        setLoadingImage2(true);
        // Simulate fetching the second image

        formData.set("img_type", "IMG2");
        axios
          .post("http://127.0.0.1:8001/api/process-image/", formData)
          .then((res) => {
            const image2Data = res.data.processed_img;
            // Display the first image
            setLoadingImage2(false);
            setImage2String(image2Data);
          });
      });

    // Hit API to fetch the second image
    // Replace 'api/image2' with your actual backend API endpoint
  };

  return (
    <div className="appWrapper">
      <ToastContainer />
      <div className="headerBackground"></div>
      <div className="headerWrapper">
        <div className="headerTitle">Road Sign Detection and Night Vision</div>
        <Button
          className="headerUploadButton"
          variant="outlined"
          color="warning"
          component="label"
        >
          Upload Image <CloudUploadIcon style={{ marginLeft: "0.7rem" }} />
          <input type="file" hidden onChange={handleImageUpload} />
        </Button>
        <div className="imageWrapper">
          {loadingImage1 && (
            <Skeleton
              variant="rounded"
              sx={{ bgcolor: "grey.500" }}
              width={300}
              height={300}
            />
          )}
          {loadingImage1 === false && (
            <img src={`data:image/jpeg;base64,${image1String}`} alt="1" />
          )}
          {loadingImage2 && (
            <Skeleton
              variant="rounded"
              sx={{ bgcolor: "grey.500" }}
              width={300}
              height={300}
            />
          )}
          {loadingImage2 === false && (
            <img src={`data:image/jpeg;base64,${image2String}`} alt="1" />
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;
