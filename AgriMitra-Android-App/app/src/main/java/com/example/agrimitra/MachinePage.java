package com.example.agrimitra;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

public class MachinePage extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_machine_page);

        // Open Amazon link when button is clicked
        CardView btn1 = findViewById(R.id.btnTool1);
        btn1.setOnClickListener(v -> openLink("https://www.deere.co.in/en/tractors/"));

        CardView btn2 = findViewById(R.id.btnTool2);
        btn2.setOnClickListener(v -> openLink("https://www.aajjo.com/product/avtar-agricultural-chisel-plough-in-sri-ganganagar-avtar-agritech"));

        CardView btn3 = findViewById(R.id.btnTool3);
        btn3.setOnClickListener(v -> openLink("https://www.amazon.in/Crompton-Residential-Regenerative-Anti-Drip-Manufacturers/dp/B0CZNPGM9W/ref=mp_s_a_1_1_sspa?dib=eyJ2IjoiMSJ9.ZzPJmuU4Ebfm9d5XblpkmY80fupZvi-k9P1z_CT-J531WyFq_cKFsVkzahJwVd1QF5VcCD8HC03rZxjaMCYiYCZYyqnlIo7BLRNB3Juyuazfrs4rIRa1pJ-6r88jPdN2wzd5b6NtvGmUUx_gXHBjlClEF_x1rkcQ18D71r_3fKQgif6NmO97CSLQBs-jXejyCLRoYwEZRFWIFGqAuc__3w.5A3euIQd3-YVOx7a2e7--5h7yWmEI_LZPEJ5rYPc1W4&dib_tag=se&keywords=water+pump&qid=1757781726&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9waG9uZV9zZWFyY2hfYXRm&psc=1"));

        CardView btn4 = findViewById(R.id.btnTool4);
        btn4.setOnClickListener(v -> openLink("https://www.aajjo.com/product/mild-steel-agriculture-seed-drill-in-mandsaur-patidar-sales-agency"));
        CardView btn5 = findViewById(R.id.btnTool5);
        btn5.setOnClickListener(v -> openLink("https://www.aajjo.com/product/6-feet-stainless-steel-paddy-harvester-machine-76-hp-capacity-1-acrehour-in-kamrup-metropolitan-tradeline-corporation"));
    }

    private void openLink(String url) {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
        startActivity(intent);
    }
}
