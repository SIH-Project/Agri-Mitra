package com.example.agrimitra;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.journeyapps.barcodescanner.ScanContract;
import com.journeyapps.barcodescanner.ScanOptions;

public class FertiCheckActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Start barcode scanner immediately
        startBarcodeScanner();
    }

    private void startBarcodeScanner() {
        ScanOptions options = new ScanOptions();
        options.setPrompt("Scan the fertilizer barcode");
        options.setBeepEnabled(true);
        options.setOrientationLocked(false);
        options.setCaptureActivity(ScannerActivity.class); // optional, default works too

        barcodeLauncher.launch(options);
    }

    // Activity Result Launcher for ZXing
    private final ActivityResultLauncher<ScanOptions> barcodeLauncher = registerForActivityResult(
            new ScanContract(),
            result -> {
                if(result.getContents() != null) {
                    String barcode = result.getContents().trim();
                    Intent intent=new Intent(FertiCheckActivity.this,FertiCheckResult.class);

                    if(barcode.startsWith("abc")) {
                        Toast.makeText(this, "Valid Fertilizer", Toast.LENGTH_LONG).show();
                        intent.putExtra("result","valid");
                        startActivity(intent);

                    } else {
                        Toast.makeText(this, "Invalid Fertilizer", Toast.LENGTH_LONG).show();
                        intent.putExtra("result","invalid");
                        startActivity(intent);

                    }

                    // Optional: restart scanner automatically
                    // startBarcodeScanner();
                } else {
                    Toast.makeText(this, "Scan cancelled", Toast.LENGTH_SHORT).show();
                    finish(); // close activity if scan cancelled
                }
            }
    );
}
