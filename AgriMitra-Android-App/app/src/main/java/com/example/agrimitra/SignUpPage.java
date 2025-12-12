package com.example.agrimitra;

import android.os.Bundle;
import android.view.View;
import android.widget.EditText;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.firebase.database.FirebaseDatabase;

public class SignUpPage extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_sign_up_page);

        EditText name=findViewById(R.id.etName);
        EditText email=findViewById(R.id.etEmail);
        EditText password=findViewById(R.id.etPassword);
        EditText state=findViewById(R.id.etFarm);
        EditText citylabel=findViewById(R.id.cityinput);
        EditText phone=findViewById(R.id.phoneinput);


        CardView submit=findViewById(R.id.btnSignUp);

        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                String key=System.currentTimeMillis()+"";

                FirebaseDatabase.getInstance().getReference().child("userid").child(key).child("name").setValue(name.getText().toString());
                FirebaseDatabase.getInstance().getReference().child("userid").child(key).child("email").setValue(email.getText().toString());
                FirebaseDatabase.getInstance().getReference().child("userid").child(key).child("password").setValue(password.getText().toString());
                FirebaseDatabase.getInstance().getReference().child("userid").child(key).child("state").setValue(state.getText().toString());
                FirebaseDatabase.getInstance().getReference().child("userid").child(key).child("citylabel").setValue(citylabel.getText().toString());
                FirebaseDatabase.getInstance().getReference().child("userid").child(key).child("phone").setValue(phone.getText().toString());

                name.setText("");
                email.setText("");
                password.setText("");
                state.setText("");
                citylabel.setText("");
                phone.setText("");



            }
        });



        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
}