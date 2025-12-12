package com.example.agrimitra;



import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import androidx.appcompat.app.AppCompatActivity;

public class LoadingActivity extends AppCompatActivity {

    private static final int SPLASH_TIME = 2000; // 2 seconds

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_loading); // We'll create this layout next

        // Get yield values from intent
        Intent intent = getIntent();
        double yieldPerAcre = intent.getDoubleExtra("yieldperacre", 0);
        double totalYield = intent.getDoubleExtra("totalyield", 0);

        // Wait 5 seconds, then start SimulatorResultPage
        new Handler().postDelayed(() -> {
            Intent resultIntent = new Intent(LoadingActivity.this, SimulatorResultPage.class);
            resultIntent.putExtra("yieldperacre", yieldPerAcre);
            resultIntent.putExtra("totalyield", totalYield);
            startActivity(resultIntent);
            finish();
        }, SPLASH_TIME);
    }
}
