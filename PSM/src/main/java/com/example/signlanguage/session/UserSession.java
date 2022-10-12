package com.example.signlanguage.session;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.WebSocketSession;


import java.net.http.WebSocket;
import java.util.ArrayList;
import java.util.List;

@Component
public class UserSession {
    private final List<WebSocketSession> session = new ArrayList<WebSocketSession>();

    public List<WebSocketSession> returnSession(){
        return session;
    }
    public void add(WebSocketSession user){
        session.add(user);
    }
    public void remove(WebSocketSession user){
        session.remove(user);
    }
}
