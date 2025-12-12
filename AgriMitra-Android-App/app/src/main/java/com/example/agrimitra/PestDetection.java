package com.example.agrimitra;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

import okhttp3.*;
import org.json.JSONObject;

public class PestDetection extends AppCompatActivity {

    private TranslationHelper translationHelper;

    private CardView btnChooseImage, btnSubmitImage;
    private ImageView imageSelectedCrop;
    private Uri selectedImageUri;

    private ActivityResultLauncher<String> imagePickerLauncher;

    private final String BASE_URL = "https://pest-detection-model-4.onrender.com";
    private final String ENDPOINT = "/api/pest/predict";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_pest_detection);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);

        btnChooseImage = findViewById(R.id.btnChooseImage);
        btnSubmitImage = findViewById(R.id.btnSubmitImage);
        imageSelectedCrop = findViewById(R.id.img2);

        // Register the image picker launcher
        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        selectedImageUri = uri;
                        imageSelectedCrop.setVisibility(ImageView.VISIBLE);
                        imageSelectedCrop.setImageURI(uri);
                        translationHelper.showTranslatedToast("Image Selected");
                    } else {
                        translationHelper.showTranslatedToast("No Image Selected");
                    }
                }
        );

        btnChooseImage.setOnClickListener(v -> imagePickerLauncher.launch("image/*"));

        // Submit button click -> send image to API
        btnSubmitImage.setOnClickListener(v -> {
            if (selectedImageUri != null) {
                sendImageForPredictionAndTransfer(selectedImageUri);
            } else {
                translationHelper.showTranslatedToast("Please select an image first");
            }
        });

        // Handle system bars
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    // Convert URI to File
    private File uriToFile(Uri uri, String fileName) throws Exception {
        InputStream inputStream = getContentResolver().openInputStream(uri);
        File file = new File(getCacheDir(), fileName);
        OutputStream outputStream = new FileOutputStream(file);
        byte[] buf = new byte[1024];
        int len;
        while ((len = inputStream.read(buf)) > 0) {
            outputStream.write(buf, 0, len);
        }
        outputStream.close();
        inputStream.close();
        return file;
    }

    private void sendImageForPredictionAndTransfer(Uri imageUri) {
        new Thread(() -> {
            try {
                // Convert URI to cache file
                File file = uriToFile(imageUri, "pest_image.jpg");

                OkHttpClient client = new OkHttpClient();

                RequestBody fileBody = RequestBody.create(file, MediaType.parse("image/*"));
                MultipartBody requestBody = new MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart("image", file.getName(), fileBody)
                        .build();

                Request request = new Request.Builder()
                        .url(BASE_URL + ENDPOINT)
                        .post(requestBody)
                        .build();

                Response response = client.newCall(request).execute();
                if (response.isSuccessful()) {
                    String resString = response.body().string();
                    JSONObject json = new JSONObject(resString);
                    String predictedPest = json.getJSONObject("prediction").getString("predicted_pest");

                    runOnUiThread(() -> {
                        // Show prediction toast
                        translationHelper.showTranslatedToast("Predicted Pest: " + predictedPest, Toast.LENGTH_LONG);

                        // Transfer to YieldPredictionResult.class
                        Intent intent = new Intent(PestDetection.this, PestDiseaseResult.class);
                        intent.putExtra("image_path", file.getAbsolutePath()); // send temp file path
                        intent.putExtra("predicted_pest", predictedPest);      // send prediction
                        startActivity(intent);
                    });
                } else {
                    runOnUiThread(() -> translationHelper.showTranslatedToast("API Error: " + response.code()));
                }
            } catch (Exception e) {
                e.printStackTrace();
                runOnUiThread(() -> translationHelper.showTranslatedToast("Error: " + e.getMessage()));
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}
