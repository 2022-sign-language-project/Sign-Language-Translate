package com.example.signlanguage.chat;

import com.example.signlanguage.chatDTO.ChatRoomDTO;
import com.example.signlanguage.manager.DBManager;
import org.bson.Document;
import org.springframework.stereotype.Repository;

import javax.annotation.PostConstruct;
import java.util.*;

@Repository
public class ChatRoomRepository {

    private Map<String, ChatRoomDTO> chatRoomDTOMap;
    private DBManager dbManager;
    @PostConstruct
    private void init(){
        chatRoomDTOMap = new LinkedHashMap<>();
        dbManager=new DBManager();
    }

    public List<Document> findAllRooms(){
//        List<ChatRoomDTO> result = new ArrayList<>(chatRoomDTOMap.values());
        //채팅방 생성 순서 최근 순으로 반환
        // Collections.reverse(result);
        List<Document> result = dbManager.findAll();

        return result;
    }

    public Document findRoomById(String id){
//        return chatRoomDTOMap.get(id);
        return dbManager.findById(id);
    }

    public ChatRoomDTO createChatRoomDTO(String name){
        ChatRoomDTO room = ChatRoomDTO.create(name);
//        chatRoomDTOMap.put(room.getRoomId(), room);

        dbManager.insert(name);
        return room;
    }
}
