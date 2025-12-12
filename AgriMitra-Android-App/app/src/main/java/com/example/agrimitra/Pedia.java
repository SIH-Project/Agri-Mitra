package com.example.agrimitra;

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

public class Pedia extends AppCompatActivity {

    private TextView
            it1, it2, it3, it4, it5, it6, it7, it8, it9, it10,
            it11, it12, it13, it14, it15, it16, it17, it18, it19, it20,
            it21, it22, it23, it24;

    TextView t1,t2,t3,t4;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_pedia);


        TranslationHelper translationHelper=new TranslationHelper(this);


        t1=findViewById(R.id.t1);
        t2=findViewById(R.id.t2);
        t3=findViewById(R.id.t3);
        t4=findViewById(R.id.t4);



        it1 = findViewById(R.id.it1);


        it2 = findViewById(R.id.it2);
        it3 = findViewById(R.id.it3);
        it4 = findViewById(R.id.it4);
        it5 = findViewById(R.id.it5);
        it6 = findViewById(R.id.it6);
        it7 = findViewById(R.id.it7);
        it8 = findViewById(R.id.it8);
        it9 = findViewById(R.id.it9);
        it10 = findViewById(R.id.it10);
        it11 = findViewById(R.id.it11);
        it12 = findViewById(R.id.it12);
        it13 = findViewById(R.id.it13);
        it14 = findViewById(R.id.it14);
        it15 = findViewById(R.id.it15);
        it16 = findViewById(R.id.it16);
        it17 = findViewById(R.id.it17);
        it18 = findViewById(R.id.it18);
        it19 = findViewById(R.id.it19);
        it20 = findViewById(R.id.it20);
        it21 = findViewById(R.id.it21);
        it22 = findViewById(R.id.it22);
        it23 = findViewById(R.id.it23);
        it24 = findViewById(R.id.it24);



        CardView c1=findViewById(R.id.c1);
        CardView c2=findViewById(R.id.c2);
        CardView c3=findViewById(R.id.c3);
        CardView c4=findViewById(R.id.c4);
        CardView c5=findViewById(R.id.c5);
        CardView c6=findViewById(R.id.c6);
        CardView c7=findViewById(R.id.c7);
        CardView c8=findViewById(R.id.c8);
        CardView c9=findViewById(R.id.c9);
        CardView c10=findViewById(R.id.c10);
        CardView c11=findViewById(R.id.c11);
        CardView c12=findViewById(R.id.c12);


        c1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Compost");
                startActivity(intent);
            }
        });


        c2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Animal Manure");
                startActivity(intent);
            }
        });

        c3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Green Manure");
                startActivity(intent);
            }
        });


        c4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Neem Oil");
                startActivity(intent);
            }
        });

        c5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Pyr");
                startActivity(intent);
            }
        });

        c6.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Dia");
                startActivity(intent);
            }
        });


        c7.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Urea");
                startActivity(intent);
            }
        });

        c8.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","DAP");
                startActivity(intent);
            }
        });

        c9.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","MOP");
                startActivity(intent);
            }
        });

        c10.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Car");
                startActivity(intent);
            }
        });

        c11.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Chlor");
                startActivity(intent);
            }
        });

        c12.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent(Pedia.this, PediaDetailPage.class);
                intent.putExtra("topic","Gly");
                startActivity(intent);
            }
        });



        translationHelper.initializeTranslation(() -> {

            translationHelper.translateTextView(it1, "Compost");


            translationHelper.translateTextView(t1, "Natural Fertilizers");
            translationHelper.translateTextView(t2, "Natural Pesticides");
            translationHelper.translateTextView(t3, "Artificial Fertilizers");
            translationHelper.translateTextView(t4, "Artificial Pesticides");


            translationHelper.translateTextView(it2, "Decomposed organic matter from plant residues, kitchen scraps, and animal waste.");
            translationHelper.translateTextView(it3, "Animal Manure");
            translationHelper.translateTextView(it4, "Waste from animals like cows, goats, poultry, and horses.");
            translationHelper.translateTextView(it5, "Green Manure");
            translationHelper.translateTextView(it6, "Growing specific crops (legumes or cowpea) and plowing them back into the soil.");

            // Natural Pesticides
            translationHelper.translateTextView(it7, "Neem Oil");
            translationHelper.translateTextView(it8, "A broad spectrum pesticide derived from neem tree seeds that disrupts insect growth and feeding.");
            translationHelper.translateTextView(it9, "Pyrethrin");
            translationHelper.translateTextView(it10, "A fast acting insecticide derived from chrysanthemum flowers.");
            translationHelper.translateTextView(it11, "Diatomaceous");
            translationHelper.translateTextView(it12, "A mechanical insect killer made from fossilized algae that dehydrates pests.");

            // Artificial Fertilizers
            translationHelper.translateTextView(it13, "Urea");
            translationHelper.translateTextView(it14, "High nitrogen fertilizer used for boosting leafy crop growth.");
            translationHelper.translateTextView(it15, "DAP");
            translationHelper.translateTextView(it16, "Popular phosphate fertilizer providing both nitrogen and phosphorus.");
            translationHelper.translateTextView(it17, "MOP");
            translationHelper.translateTextView(it18, "Potassium chloride fertilizer that enhances drought resistance and yield.");

            // Artificial Pesticides
            translationHelper.translateTextView(it19, "Carbaryl");
            translationHelper.translateTextView(it20, "Broad-spectrum insecticide effective against many chewing and sucking pests.");
            translationHelper.translateTextView(it21, "Chlorpyrifos");
            translationHelper.translateTextView(it22, "Organophosphate pesticide commonly used for soil and foliage pests.");
            translationHelper.translateTextView(it23, "Glyphosate");
            translationHelper.translateTextView(it24, "Widely used herbicide for controlling weeds in crop fields.");


        });





























        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
}