package com.example.agrimitra;

import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class RetrofitClient {
    private static Retrofit retrofit;

    public static Retrofit getClient(String baseUrl) {
        return new Retrofit.Builder()
                .baseUrl(baseUrl) // dummy required, overridden by @Url
                .addConverterFactory(GsonConverterFactory.create())
                .build();
    }
}

