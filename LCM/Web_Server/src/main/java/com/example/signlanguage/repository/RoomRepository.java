package com.example.signlanguage.repository;

import com.example.signlanguage.chatDTO.ChatRoomDTO;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface RoomRepository extends MongoRepository<ChatRoomDTO, String> {

}
