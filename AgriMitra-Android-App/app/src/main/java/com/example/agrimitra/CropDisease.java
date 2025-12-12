package com.example.agrimitra;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class CropDisease extends AppCompatActivity {

    private TranslationHelper translationHelper;

    private Spinner spinnerCrop;
    private CardView btnChooseImage, submit;
    private ImageView imageSelectedCrop;
    private TextView cropSelectionLabel, chooseImageText, submitText;
    private Uri selectedImageUri;

    private ActivityResultLauncher<String> imagePickerLauncher;

    private static final String BASE_URL = "https://crop-disease-detection-6.onrender.com/";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_crop_disease);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);

        // Find views
        spinnerCrop = findViewById(R.id.spinnerCrop);
        btnChooseImage = findViewById(R.id.btnChooseImage);
        submit = findViewById(R.id.btnSubmitImage);
        imageSelectedCrop = findViewById(R.id.imageSelectedCrop);
        cropSelectionLabel = findViewById(R.id.tvSelectCrop);
        chooseImageText = findViewById(R.id.select_image);
        submitText = findViewById(R.id.submit_image);

        // Initialize translation and spinner
        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(cropSelectionLabel, "Select Crop");
            translationHelper.translateTextView(chooseImageText, "Select Image");
            translationHelper.translateTextView(submitText, "Submit");

            // Setup spinner
            String[] crops = {"Select Crop Type","Rice", "Cotton", "Sugarcane"};
            ArrayAdapter<String> adapter = new ArrayAdapter<>(
                    this,
                    R.layout.spinner_dropdown_item,
                    crops
            );
            adapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
            spinnerCrop.setAdapter(adapter);
        });

        // Register image picker
        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        selectedImageUri = uri;
                        imageSelectedCrop.setVisibility(ImageView.VISIBLE);
                        imageSelectedCrop.setImageURI(uri);
                        translationHelper.showTranslatedToast("Image Selected");
                    }
                }
        );

        // Image button click
        btnChooseImage.setOnClickListener(v -> imagePickerLauncher.launch("image/*"));

        // Submit button click
        submit.setOnClickListener(v -> {
            if (selectedImageUri == null) {
                translationHelper.showTranslatedToast("Please select an image first");
                return;
            }
            uploadImage();
        });

        // Handle insets
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void uploadImage() {
        String crop = spinnerCrop.getSelectedItem().toString();
        String url = "";
        String baseUrlForRetrofit = BASE_URL;

        if (crop.equalsIgnoreCase("Cotton")) {
            url = BASE_URL + "api/cotton/predict";
        } else if (crop.equalsIgnoreCase("Sugarcane")) {
            url = BASE_URL + "api/sugarcane/predict";
        } else if (crop.equalsIgnoreCase("Rice")) {
            url = "https://rice-disease-detection-4.onrender.com/predict";
            baseUrlForRetrofit = "https://rice-disease-detection-4.onrender.com/";
        } else {
            translationHelper.showTranslatedToast("No API available for " + crop);
            return;
        }

        File file = uriToFile(selectedImageUri);
        if (file == null) {
            translationHelper.showTranslatedToast("Error converting image");
            return;
        }

        RequestBody requestFile = RequestBody.create(MediaType.parse("image/*"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("image", file.getName(), requestFile);

        ApiService apiService = RetrofitClient.getClient(baseUrlForRetrofit).create(ApiService.class);
        Call<ResponseBody> call = apiService.uploadImage(url, body);

        call.enqueue(new Callback<ResponseBody>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                if (response.isSuccessful() && response.body() != null) {
                    try {
                        String json = response.body().string();
                        JSONObject jsonObject = new JSONObject(json);
                        String predictedClass;
                        if (jsonObject.has("best_prediction")) {
                            predictedClass = jsonObject.getJSONObject("best_prediction").getString("class");
                        } else {
                            predictedClass = jsonObject.getString("predicted_class");
                        }
                        Intent intent = new Intent(CropDisease.this, CropDiseaseResult.class);
                        intent.putExtra("crop", predictedClass);
                        startActivity(intent);
                    } catch (Exception e) {
                        e.printStackTrace();
                        translationHelper.showTranslatedToast("Error parsing response");
                    }
                } else {
                    translationHelper.showTranslatedToast("Upload failed: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                translationHelper.showTranslatedToast("Error: " + t.getMessage());
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }

    private File uriToFile(Uri uri) {
        try {
            File file = new File(getCacheDir(), "upload.png");
            InputStream inputStream = getContentResolver().openInputStream(uri);
            FileOutputStream outputStream = new FileOutputStream(file);
            byte[] buffer = new byte[1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            outputStream.close();
            inputStream.close();
            return file;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
