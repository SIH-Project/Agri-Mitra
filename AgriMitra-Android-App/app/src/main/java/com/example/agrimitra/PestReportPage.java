package com.example.agrimitra;

import static androidx.core.content.ContentProviderCompat.requireContext;

import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class PestReportPage extends AppCompatActivity {

    private TranslationHelper translationHelper;

    Spinner spinnerPestType, spinnerDistrict;
    CardView btnSubmit;
    ImageView imagePest;

    String[] pestTypes = {
            "Select Pest",
            "Potato Tuber Moth",
            "Aphids",
            "Whiteflies",
            "Leafhoppers",
            "Cutworms",
            "Termites",
            "Flea Beetles",
            "Colorado Potato Beetle"
    };

    String chennai,coimbatore,madurai,trichy,salem,erode,tirunelveli,vellore,thanjavur;
    int chennai_int,coimbatore_int,madurai_int,trichy_int,salem_int,erode_int,tirunelveli_int,vellore_int,thanjavur_int;

    String[] districts = {
            "Select District", "Chennai", "Coimbatore", "Madurai", "Trichy",
            "Salem", "Erode", "Tirunelveli", "Vellore", "Thanjavur"
    };



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_pest_report_page);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);





        DatabaseReference databaseReference1 = FirebaseDatabase.getInstance().getReference("reports").child("chennai");
        DatabaseReference databaseReference2 = FirebaseDatabase.getInstance().getReference("reports").child("coimbatore");
        DatabaseReference databaseReference3 = FirebaseDatabase.getInstance().getReference("reports").child("madurai");
        DatabaseReference databaseReference4 = FirebaseDatabase.getInstance().getReference("reports").child("trichy");
        DatabaseReference databaseReference5 = FirebaseDatabase.getInstance().getReference("reports").child("salem");
        DatabaseReference databaseReference6 = FirebaseDatabase.getInstance().getReference("reports").child("erode");
        DatabaseReference databaseReference7 = FirebaseDatabase.getInstance().getReference("reports").child("tirunelveli");
        DatabaseReference databaseReference8 = FirebaseDatabase.getInstance().getReference("reports").child("vellore");
        DatabaseReference databaseReference9 = FirebaseDatabase.getInstance().getReference("reports").child("thanjavur");


        databaseReference1.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    chennai=snapshot.getValue().toString();
                    chennai_int=Integer.parseInt(chennai);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


        databaseReference2.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    coimbatore=snapshot.getValue().toString();
                    coimbatore_int=Integer.parseInt(coimbatore);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


        databaseReference3.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    madurai=snapshot.getValue().toString();
                    madurai_int=Integer.parseInt(madurai);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


        databaseReference4.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    trichy=snapshot.getValue().toString();
                    trichy_int=Integer.parseInt(trichy);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


        databaseReference5.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    salem=snapshot.getValue().toString();
                    salem_int=Integer.parseInt(salem);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


        databaseReference6.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    erode=snapshot.getValue().toString();
                    erode_int=Integer.parseInt(erode);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


        databaseReference7.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    tirunelveli=snapshot.getValue().toString();
                    tirunelveli_int=Integer.parseInt(tirunelveli);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });



        databaseReference8.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    vellore=snapshot.getValue().toString();
                    vellore_int=Integer.parseInt(vellore);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });


        databaseReference9.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (snapshot.exists())
                {
                    thanjavur=snapshot.getValue().toString();
                    thanjavur_int=Integer.parseInt(thanjavur);

                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });





        // Initialize UI elements
        spinnerPestType = findViewById(R.id.spinnerPestType);
        spinnerDistrict = findViewById(R.id.spinnerDistrict);
        btnSubmit = findViewById(R.id.btnSubmit);
        imagePest = findViewById(R.id.imagePest);

        imagePest.setVisibility(View.GONE);

        // Initialize translation and set up spinners
        translationHelper.initializeTranslation(() -> {
            // Pest Type Dropdown
            ArrayAdapter<String> pestAdapter = new ArrayAdapter<>(
                    this,
                    R.layout.spinner_item,
                    pestTypes
            );

            pestAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
            spinnerPestType.setAdapter(pestAdapter);

            // District Dropdown
            ArrayAdapter<String> districtAdapter = new ArrayAdapter<>(
                    this,
                    R.layout.spinner_item,
                    districts
            );

            districtAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
            spinnerDistrict.setAdapter(districtAdapter);
        });

        spinnerPestType.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                if (position == 0) {
                    // User has not selected a pest
                    imagePest.setVisibility(View.GONE);
                } else {
                    switch (position) {
                        case 1: // Potato Tuber Moth
                            imagePest.setImageResource(R.drawable.pest1);
                            break;
                        case 2: // Aphids
                            imagePest.setImageResource(R.drawable.pest2);
                            break;
                        case 3: // Whiteflies
                            imagePest.setImageResource(R.drawable.pest3);
                            break;
                        case 4: // Jassids
                            imagePest.setImageResource(R.drawable.pest4);
                            break;
                        case 5: // Cutworms
                            imagePest.setImageResource(R.drawable.pest5);
                            break;
                        case 6: // Termites
                            imagePest.setImageResource(R.drawable.pest6);
                            break;
                        case 7: // Flea Beetles
                            imagePest.setImageResource(R.drawable.pest7);
                            break;
                        case 8: // Colorado Potato Beetle
                            imagePest.setImageResource(R.drawable.pest8);
                            break;
                    }
                    imagePest.setVisibility(View.VISIBLE);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                imagePest.setVisibility(View.GONE);
            }
        });


        // Button click logic
        btnSubmit.setOnClickListener(v -> {
            String selectedPest = spinnerPestType.getSelectedItem().toString();
            String selectedDistrict = spinnerDistrict.getSelectedItem().toString();

            if (selectedPest.equals("Select Pest") || selectedDistrict.equals("Select District")) {
                translationHelper.showTranslatedToast("Please select pest and district");
                return;
            }

            translationHelper.showTranslatedToast(
                    "Pest: " + selectedPest + "\nDistrict: " + selectedDistrict,
                    Toast.LENGTH_LONG);

            String[] districts = {
                    "Select District", "Chennai", "Coimbatore", "Madurai", "Trichy",
                    "Salem", "Erode", "Tirunelveli", "Vellore", "Thanjavur"
            };


            if (selectedDistrict.equals("Chennai"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("chennai").setValue(chennai_int+1);


            } else if (selectedDistrict.equals("Coimbatore"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("coimbatore").setValue(coimbatore_int+1);


            } else if (selectedDistrict.equals("Madurai"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("madurai").setValue(madurai_int+1);


            } else if (selectedDistrict.equals("Trichy"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("trichy").setValue(trichy_int+1);


            } else if (selectedDistrict.equals("Salem"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("salem").setValue(salem_int+1);


            } else if (selectedDistrict.equals("Erode"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("erode").setValue(erode_int+1);


            } else if (selectedDistrict.equals("Tirunelveli"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("tirunelveli").setValue(tirunelveli_int+1);


            } else if (selectedDistrict.equals("Vellore"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("vellore").setValue(vellore_int+1);


            } else if (selectedDistrict.equals("Thanjavur"))
            {
                FirebaseDatabase.getInstance().getReference("reports").child("thanjavur").setValue(thanjavur_int+1);
            }




        });






        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}