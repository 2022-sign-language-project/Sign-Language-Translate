# SignLanguage-Translate-Web
졸업 프로젝트

### MongoDB 설치하기
https://kitty-geno.tistory.com/155 

### Web_Server/build.gradle 추가하기
필요한 모듈 설치
```
	// mongodb
	implementation 'org.springframework.data:spring-data-mongodb:3.4.2'
	implementation 'org.springframework.boot:spring-boot-starter'
	implementation 'org.springframework.boot:spring-boot-devtools:2.7.3'
	implementation 'org.springframework.boot:spring-boot-starter-data-mongodb:2.7.3'

	testImplementation 'org.springframework.boot:spring-boot-starter-test'
```

 ### Web_Server/src/main/resources/application.properties 추가하기
 mongodb 와 연결하는 부분
``` 
 # Mongo Configuration
server.port:8080
spring.data.mongodb.host=localhost
spring.data.mongodb.port=27017
spring.data.mongodb.database=Chat
```

### Web_Server/src/main/java/com/example/signlanguage
변경되거나 추가된 파일
- chatDTO/chatRoomDTO.java: 생성자 추가
- repository/RoomRepository.java: MongoRepository 상속 
- controller/RoomController.java: 각 메소드마다 RoomRepository를 적용
