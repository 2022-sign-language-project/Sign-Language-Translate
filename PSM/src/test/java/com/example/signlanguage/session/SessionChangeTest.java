package com.example.signlanguage.session;

import com.example.signlanguage.AppConfig;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.web.socket.WebSocketSession;

import java.util.List;


public class SessionChangeTest {
    @Test
    @DisplayName("sessoin list가 usersession이라는 class로 대체가 되나?")
    public void test(){
        AnnotationConfigApplicationContext ac = new AnnotationConfigApplicationContext(AppConfig.class);
        UserSession us = ac.getBean(UserSession.class);

        List<WebSocketSession> wss = us.returnSession();
        System.out.println(wss);
    }
}
