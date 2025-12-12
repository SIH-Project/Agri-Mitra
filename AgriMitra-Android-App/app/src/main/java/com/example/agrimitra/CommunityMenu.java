package com.example.agrimitra;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class CommunityMenu extends AppCompatActivity {

    TextView t0,t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_community_menu);

        CardView c1=findViewById(R.id.c1);
        CardView c2=findViewById(R.id.c2);
        CardView c3=findViewById(R.id.c3);
        CardView c4=findViewById(R.id.c4);
        CardView c5=findViewById(R.id.c5);

        TranslationHelper translationHelper=new TranslationHelper(this);


        t1 = findViewById(R.id.t1);
        t0 = findViewById(R.id.t0);
        t2 = findViewById(R.id.t2);
        t3 = findViewById(R.id.t3);
        t4 = findViewById(R.id.t4);
        t5 = findViewById(R.id.t5);
        t6 = findViewById(R.id.t6);
        t7 = findViewById(R.id.t7);
        t8 = findViewById(R.id.t8);
        t9 = findViewById(R.id.t9);
        t10 = findViewById(R.id.t10);


        translationHelper.initializeTranslation(() -> {
            if(t1 != null) translationHelper.translateTextView(t1, "Explore Communities");
            if(t0 != null) translationHelper.translateTextView(t0, "Farmers of Maharashtra");

            if(t2 != null) translationHelper.translateTextView(t2, "Connect with farmers in Maharashtra");
            if(t3 != null) translationHelper.translateTextView(t3, "Farmers of TamilNadu");
            if(t4 != null) translationHelper.translateTextView(t4, "Connect with farmers in TamilNadu");

            if(t5 != null) translationHelper.translateTextView(t5, "Farmers of UP");
            if(t6 != null) translationHelper.translateTextView(t6, "Connect with farmers in UP");

            if(t7 != null) translationHelper.translateTextView(t7, "Farmers of Kerala");
            if(t8 != null) translationHelper.translateTextView(t8, "Connect with farmers in Kerala");

            if(t9 != null) translationHelper.translateTextView(t9, "Farmers of Karnataka");
            if(t10 != null) translationHelper.translateTextView(t10, "Connect with farmers in Karnataka");
        });

        c1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(CommunityMenu.this, CommunityChat.class);
                intent.putExtra("community","MH");
                startActivity(intent);
            }
        });

        c2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(CommunityMenu.this, CommunityChat.class);
                intent.putExtra("community","TN");
                startActivity(intent);
            }
        });

        c3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(CommunityMenu.this, CommunityChat.class);
                intent.putExtra("community","UP");
                startActivity(intent);
            }
        });

        c4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(CommunityMenu.this, CommunityChat.class);
                intent.putExtra("community","K");
                startActivity(intent);
            }
        });

        c5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(CommunityMenu.this, CommunityChat.class);
                intent.putExtra("community","KAR");
                startActivity(intent);
            }
        });









        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
}