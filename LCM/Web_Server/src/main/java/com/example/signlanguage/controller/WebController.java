package com.example.signlanguage.controller;

import com.example.signlanguage.repository.RoomRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.servlet.ModelAndView;

@Controller
@RequiredArgsConstructor
public class WebController {
    private final RoomRepository repository;

    @GetMapping(value = "/hospital")
    public ModelAndView hospital() {
        System.out.println("Chat Room Created!!!!");
        ModelAndView mv = new ModelAndView("hospital/hospital");

        mv.addObject("list", repository.findAll());

        return mv;
        // return "hospital/hospital";
    }
}
