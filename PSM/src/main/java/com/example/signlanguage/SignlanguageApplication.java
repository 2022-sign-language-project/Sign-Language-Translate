package com.example.signlanguage;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;

// exclude는 일단 지금은 DB를 사용하지 않기 때문에 사용함
@SpringBootApplication(exclude={DataSourceAutoConfiguration.class})
public class SignlanguageApplication {

	public static void main(String[] args) {
		SpringApplication.run(SignlanguageApplication.class, args);
	}

}
