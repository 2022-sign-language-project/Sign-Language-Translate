package com.example.signlanguage.controller;

import com.example.signlanguage.chatDTO.ChatRoomDTO;
import com.example.signlanguage.repository.RoomRepository;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.data.mongo.DataMongoTest;
import org.springframework.test.annotation.Rollback;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

//@ExtendWith(SpringExtension.class)
//@ContextConfiguration(locations = {"file:/resources/application.properties"})
//@Transactional
@Rollback
@RunWith(SpringRunner.class)
@DataMongoTest
public class RoomControllerTest {

    @Autowired
    private RoomRepository repository;
    private String randomString() {
        String id = "";
        for (int i = 0; i < 10; i++) {
            double dValue = Math.random();
            if (i % 2 == 0) {
                id += (char) ((dValue * 26) + 65);   // 대문자
                continue;
            }
            id += (char) ((dValue * 26) + 97); // 소문자
        }
        return id;
    }
    @Test
    void getRooms() {
        final String NAME=randomString();
        final int INSERT_SIZE=5;
        insertFindAllTestData(NAME,INSERT_SIZE );

        List<ChatRoomDTO> findRooms=repository.findAll();

        assertEquals(findRooms.size(), INSERT_SIZE);
    }
    void insertFindAllTestData(String NAME, int INSERT_SIZE) {
        for (int i = 0; i < INSERT_SIZE; ++i) {
            ChatRoomDTO room = ChatRoomDTO.builder().roomId(Integer.toString(i*100)).name(NAME).build();
            repository.save(room);
        }
    }
    @Test
    void saveRoom() {
        ChatRoomDTO room = ChatRoomDTO.builder().roomId("12345").name("test room").build();
        repository.save(room);

        ChatRoomDTO findRoom = repository.findById(room.getRoomId()).get();

        assertEquals(room.getRoomId(), findRoom.getRoomId());
        assertEquals(room.getName(), findRoom.getName());

    }

    @Test
    void getRoom() {
        ChatRoomDTO room = ChatRoomDTO.builder().roomId("12345").name("test room").build();
        repository.save(room);

        ChatRoomDTO findRoom = repository.findById(room.getRoomId()).get();

        assertEquals(room.getRoomId(), findRoom.getRoomId());
        assertEquals(room.getName(), findRoom.getName());
    }
}