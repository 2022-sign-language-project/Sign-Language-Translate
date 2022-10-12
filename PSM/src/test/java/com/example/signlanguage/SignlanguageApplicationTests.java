package com.example.signlanguage;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

@SpringBootTest
class SignlanguageApplicationTests {


	@Test
	void contextLoads() {
		ApplicationContext ac = new AnnotationConfigApplicationContext(AppConfig.class);
		if (ac != null){
			String[] beanDefinitionNames = ac.getBeanDefinitionNames();

			for (String beanDefinitionName : beanDefinitionNames) {
				System.out.println(beanDefinitionName);
			}
		}
	}

}
