package com.example.signlanguage.Rest;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.converter.FormHttpMessageConverter;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.http.converter.StringHttpMessageConverter;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;

@RestController
public class RestAPICall implements RestAPI{
    @Override
//    @RequestMapping("/test")
    public String Get(@RequestParam String str) {
        return str+ "Test 됨";
    }

    @Override
    @RequestMapping("/get_prediction")
    public void GetCall() {
//        // RestTemplate 에 MessageConverter 세팅
//        List<HttpMessageConverter<?>> converters = new ArrayList<HttpMessageConverter<?>>();
//        converters.add(new FormHttpMessageConverter());
//        converters.add(new StringHttpMessageConverter());

        final String url = "http://127.0.0.1:5500/test";
        RestTemplate restTemplate = new RestTemplate();

        // header
        HttpHeaders httpHeaders = new HttpHeaders();
        HttpEntity<?> entity = new HttpEntity<>(httpHeaders);
        // httpHeaders.setContentType(MediaType.APPLICATION_JSON);

        // Body
//        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
//        body.add("prediction", "this is test");

        // Message
        HttpEntity<?> requestMessage = new HttpEntity<>(httpHeaders);

        // request
        HttpEntity<?> response = restTemplate.exchange(url, HttpMethod.GET,entity, Object.class);
//        HttpEntity<String> response = restTemplate.postForEntity(url, requestMessage, String.class);

        // response parsing
        ObjectMapper objectMapper = new ObjectMapper();
        // objectMapper.configure(DeserializationFeature.ACCEPT_EMPTY_STRING_AS_NULL_OBJECT, true);
//        System.out.println(response.getStatusCodeValue());
        System.out.println(response);
        System.out.println(response.getBody());
    }


    @Override
    public void Post() {

    }
}
