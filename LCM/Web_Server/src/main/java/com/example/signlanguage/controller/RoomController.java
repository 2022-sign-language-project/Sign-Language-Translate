package com.example.signlanguage.controller;

import com.example.signlanguage.chatDTO.ChatRoomDTO;
import com.example.signlanguage.repository.RoomRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
@RequiredArgsConstructor
@RequestMapping(value = "/chat")
//@RestController
public class RoomController {
    @Autowired
    private final RoomRepository repository;

    //채팅방 목록 조회
    @GetMapping(value = "/rooms")
    public ModelAndView getRooms(){

        ModelAndView mv = new ModelAndView("chat/rooms");
        mv.addObject("list", repository.findAll());
        return mv;
    }

    //채팅방 개설
    @PostMapping(value = "/room")
    public String saveRoom(@RequestParam String name, RedirectAttributes rttr){

        ChatRoomDTO room = new ChatRoomDTO(name);
        rttr.addFlashAttribute("roomName", repository.save(room));
        return "redirect:/chat/rooms";
    }

    //채팅방 조회
    @GetMapping("/room")
    public void getRoom(String roomId, Model model){
        ChatRoomDTO room = repository.findById(roomId).get();
        model.addAttribute("room", room);
    }

    @DeleteMapping("/delete")
    public String deleteAllRooms(){
        repository.deleteAll();
        return "Deleted All";
    }

//    @PostMapping(value = "/addRoom")
//    public String saveRoom(@RequestBody ChatRoomDTO room){
//        repository.save(room);
//        return "Added";
//    }
//    @GetMapping("/findAllRooms")
//    public List<ChatRoomDTO> getRooms(){
//        return repository.findAll();
//    }
//    @GetMapping("/find/{id}")
//    public Optional<ChatRoomDTO> getRoom(@PathVariable String id){
//        return repository.findById(id);
//    }
//    @DeleteMapping("/delete/{id}")
//    public String deleteRoom(@PathVariable String id){
//        repository.deleteById(id);
//        return "Deleted";
//    }
}
