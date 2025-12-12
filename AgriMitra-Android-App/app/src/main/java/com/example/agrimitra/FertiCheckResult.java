package com.example.agrimitra;

import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.w3c.dom.Text;

public class FertiCheckResult extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {


        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_ferti_check_result);


        ImageView imageView=findViewById(R.id.image_result);
        TextView wish=findViewById(R.id.wish);
        TextView wish2=findViewById(R.id.wish2);

        String result=getIntent().getStringExtra("result").toString();

        if (result.equals("valid"))
        {
            imageView.setImageResource(R.drawable.valid);
            wish.setText("Valid Fertilizer");
            wish2.setText("Government(Department of Fertilizers) has approved this fertilizer/Pesticide");
        }
        else
        {
            imageView.setImageResource(R.drawable.invalid);
            wish.setText("Invalid Fertilizer");
            wish2.setText("Government(Department of Fertilizers) has not approved this fertilizer/Pesticide");

        }




        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
}