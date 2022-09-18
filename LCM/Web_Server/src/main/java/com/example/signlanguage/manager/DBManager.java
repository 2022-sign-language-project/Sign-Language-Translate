package com.example.signlanguage.manager;

import com.example.signlanguage.chatDTO.ChatRoomDTO;
import com.mongodb.client.*;
import org.bson.Document;

import static com.mongodb.client.model.Filters.eq;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class DBManager {
    private MongoDatabase database;
    private MongoCollection<Document> collection;

    String URI = "mongodb://localhost:27017";
    String DATABASE = "chatting_room";
    String COLLECTION = "rooms";
    public DBManager(){
        // connect mongodb
        MongoClient client = MongoClients.create(URI);
        this.database = client.getDatabase(DATABASE);
        //        database.createCollection(COLLECTION);

        this.collection=database.getCollection(COLLECTION);
    }

    public void insert(String name){
        ChatRoomDTO room = ChatRoomDTO.create(name);
        collection.insertOne(new Document("roomID",room.getRoomId())
                .append("name",name));
    }
    public List<Document> findAll(){
        Iterator iterator = collection.find().iterator();
        List<Document> result = new ArrayList<>();
        while(iterator.hasNext()){
            result.add((Document) iterator.next());
        }
        return result;
    }
    public Document findById(String id){
        Document doc = collection.find(eq("name",id)).first();
        return doc;
    }
    public void delete(Document query){
        collection.deleteOne(query);
    }


//    public static void main(String[] args) {
//
//        DBManager dbManager = new DBManager();
//
//        dbManager.insert("333");
//        List<Document> documents =dbManager.findAll();
//        for (Document d : documents) {
//            System.out.println(d);
//        }
//        Document doc =dbManager.findById("333");
//        System.out.println(doc);
//    }
}

