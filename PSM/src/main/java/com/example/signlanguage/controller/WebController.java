package com.example.signlanguage.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
@RequiredArgsConstructor
public class WebController {

    @GetMapping(value = "/hospital")
    public String hospital(){
        System.out.println("IN");
        return "hospital/hospital";
    }
}
