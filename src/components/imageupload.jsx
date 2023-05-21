import axios from "axios";
import React, { useState } from "react";
const ImageUpload = () => {
  const [uploading, setUploading] = useState(false);
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
    const base64image = await convertBase64(file);

    setUploading(true);
    setLoadingImage1(true);

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

    setUploading(false);
  };

  return (
    <div>
      {uploading && <p>Uploading image...</p>}
      {loadingImage1 && <p>Loading first image...</p>}
      {loadingImage1 === false && (
        <img src={`data:image/jpeg;base64,${image1String}`} alt="1" />
      )}
      {loadingImage2 && <p>Loading second image...</p>}
      {loadingImage2 === false && (
        <img src={`data:image/jpeg;base64,${image2String}`} alt="1" />
      )}

      <input type="file" onChange={handleImageUpload} />
    </div>
  );
};

export default ImageUpload;
