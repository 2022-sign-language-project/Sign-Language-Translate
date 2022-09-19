package com.example.signlanguage.manager;

public class DBManager {
//    private MongoDatabase database;
//    private MongoCollection<Document> collection;
//
//    String URI = "mongodb://localhost:27017";
//    String DATABASE = "chatting_room";
//    String COLLECTION = "rooms";
//    public DBManager(){
//        // connect mongodb
//        MongoClient client = MongoClients.create(URI);
//        this.database = client.getDatabase(DATABASE);
//        this.collection=database.getCollection(COLLECTION);
//    }
//
//    public void insert(ChatRoomDTO room){
//        collection.insertOne(new Document("roomID", room.getRoomId())
//                .append("name",room.getName()));
//    }
//    public List<ChatRoomDTO> findAll(){
//        Iterator iterator = collection.find().iterator();
//        List<ChatRoomDTO> result = new ArrayList<>();
//        while(iterator.hasNext()){
//            Document doc = (Document) iterator.next();
//            result.add(new ChatRoomDTO((String) doc.get("name")));
//            System.out.println("find all rooms : "+doc);
//        }
//        return result;
//    }
//    public ChatRoomDTO findById(String id){
//        Document doc = collection.find(eq("roomID",id)).first();
//        System.out.println("find by roomID : "+doc);
//
//        return new ChatRoomDTO((String) doc.get("roomID"));
//    }
//    public void delete(Document query){
//        collection.deleteOne(query);
//    }
//
//    public void create(){
//        database.createCollection(COLLECTION);
//    }
//
//    public void drop(){
//        collection.drop();
//    }
//
//    public static void main(String[] args) {
//
//        DBManager dbManager = new DBManager();
////        dbManager.drop();
////        dbManager.create();
//        ChatRoomDTO room = new ChatRoomDTO("555");
//        dbManager.insert(room);
//        dbManager.findAll();
////        for (Document d : documents) {
////            System.out.println(d);
////        }
//        dbManager.findById("333");
////        System.out.println(doc);
//    }
}

