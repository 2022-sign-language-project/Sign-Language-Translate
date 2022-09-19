package com.example.signlanguage.chatDTO;

import lombok.*;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.web.socket.WebSocketSession;

import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

@Getter
@Setter
@Data
@NoArgsConstructor
@AllArgsConstructor
@Document(collection="Rooms")
public class ChatRoomDTO {
    @Id
    private String roomId;
    private String name;
    private Set<WebSocketSession> sessions = new HashSet<>();
    //WebSocketSession은 Spring에서 Websocket Connection이 맺어진 세션

    public ChatRoomDTO(String name){
        this.roomId = UUID.randomUUID().toString();
        this.name = name;
    }

//    public static ChatRoomDTO create(String name){
//        ChatRoomDTO room = new ChatRoomDTO();
//
//        room.roomId = UUID.randomUUID().toString();
//        room.name = name;
//        return room;
//    }
}
