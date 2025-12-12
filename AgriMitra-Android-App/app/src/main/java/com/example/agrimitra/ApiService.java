package com.example.agrimitra;

import okhttp3.MultipartBody;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Url;

public interface ApiService {
    @Multipart
    @POST
    Call<ResponseBody> uploadImage(
            @Url String url, // we pass dynamic url
            @Part MultipartBody.Part image
    );
}
